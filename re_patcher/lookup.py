"""newminhh lookup table parser, signature DSL, staleness checker, and symbol resolver.

Pure-Python implementation with exact 1:1 parity with the newminhh lookup
architecture defined in src/common/newminhh_abi.h, src/common/newminhh_lookup.h,
and docs/lookup_table.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import struct
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:
    from . import PatcherError
except ImportError:
    try:
        from re_patcher import PatcherError  # type: ignore[no-redef]
    except ImportError:
        class PatcherError(Exception):  # type: ignore[no-redef]
            """Structured fail-loudly exception for patcher errors."""

            def __init__(self, message: str, exit_code: int = 1) -> None:
                super().__init__(message)
                self.message = message
                self.exit_code = exit_code


# ---------------------------------------------------------------------------
# 1. Exact Constants matching src/common/newminhh_abi.h and newminhh_lookup.h
# ---------------------------------------------------------------------------

# Return kind constants (NM_RET_*, 0..7)
NM_RET_VOID = 0
NM_RET_I32 = 1
NM_RET_U32 = 2
NM_RET_PTR = 3
NM_RET_I64 = 4
NM_RET_U64 = 5
NM_RET_F32 = 6
NM_RET_F64 = 7

# Argument kind constants (NM_ARG_*, 0..7)
NM_ARG_NONE = 0
NM_ARG_I32 = 1
NM_ARG_U32 = 2
NM_ARG_PTR = 3
NM_ARG_I64 = 4
NM_ARG_U64 = 5
NM_ARG_F32 = 6
NM_ARG_F64 = 7

NM_MAX_SIG_ARGS = 16

# nm_fn_sig_t ABI extension flags
NM_SIG_EXT_SRET = 1 << 0
NM_SIG_EXT_CALLEE_POP4 = 1 << 1


def nm_sig_ext_make_sret(n: int) -> int:
    """Build abi_ext byte with sret, callee-pop-4, and n extra stack slots."""
    return (NM_SIG_EXT_SRET | NM_SIG_EXT_CALLEE_POP4 | ((n & 0x1F) << 3)) & 0xFF


def nm_sig_ext_sret_size(e: int) -> int:
    """Extract sret stack slot count (bits 3..7) from abi_ext byte."""
    return (e >> 3) & 0x1F


NM_SIG_EXT_MAKE_SRET = nm_sig_ext_make_sret
NM_SIG_EXT_SRET_SIZE = nm_sig_ext_sret_size

# Symbol / callback flags (matching newminhh_abi.h)
NM_SYM_FLAG_DATA = 1 << 0
NM_SYM_FLAG_VARIADIC = 1 << 1
NM_SYM_FLAG_CDECL_DIRECT = 1 << 2
NM_SYM_FLAG_IFUNC = 1 << 3
NM_SYM_FLAG_NATIVE = 1 << 4
NM_SYM_FLAG_PATCHED = 1 << 5
NM_SYM_FLAG_STATIC_ONLY = 1 << 6
NM_SYM_FLAG_RUNTIME_ONLY = 1 << 7
NM_SYM_FLAG_WEAK = 1 << 8
NM_SYM_FLAG_PURE = 1 << 9

# Lookup classes (matching newminhh_lookup.h)
NM_LK_NONE = 0
NM_LK_COMPAT = 1
NM_LK_NATIVE_TYPED = 2
NM_LK_NATIVE_DATA = 3
NM_LK_PATCHED_CDECL = 4
NM_LK_ALIAS = 5
NM_LK_UNTYPED = 6
NM_LK_FORBIDDEN = 7
NM_LK_NATIVE_LIB = 8

CLASS_NAME_TO_ID = {
    "NONE": NM_LK_NONE,
    "COMPAT": NM_LK_COMPAT,
    "NATIVE": NM_LK_NATIVE_LIB,
    "NATIVE_TYPED": NM_LK_NATIVE_TYPED,
    "NATIVE_DATA": NM_LK_NATIVE_DATA,
    "PATCHED_CDECL": NM_LK_PATCHED_CDECL,
    "ALIAS": NM_LK_ALIAS,
    "UNTYPED": NM_LK_UNTYPED,
    "FORBIDDEN": NM_LK_FORBIDDEN,
    "FORBIDDEN_ABI": NM_LK_FORBIDDEN,
    "NATIVE_LIB": NM_LK_NATIVE_LIB,
}

FLAG_NAME_TO_MASK = {
    "DATA": NM_SYM_FLAG_DATA,
    "VARIADIC": NM_SYM_FLAG_VARIADIC,
    "CDECL_DIRECT": NM_SYM_FLAG_CDECL_DIRECT,
    "IFUNC": NM_SYM_FLAG_IFUNC,
    "NATIVE": NM_SYM_FLAG_NATIVE,
    "PATCHED": NM_SYM_FLAG_PATCHED,
    "STATIC_ONLY": NM_SYM_FLAG_STATIC_ONLY,
    "RUNTIME_ONLY": NM_SYM_FLAG_RUNTIME_ONLY,
    "WEAK": NM_SYM_FLAG_WEAK,
    "PURE": NM_SYM_FLAG_PURE,
}

DEFAULT_CACHE_PATH = "/var/cache/newminhh/lookup_table.txt"
DEFAULT_BASE_PATH = "/usr/share/newminhh/lookup_base.txt"
HEADER_MAGIC = "#!newminhh-lookup v1"

_RET_CHAR_MAP: Dict[str, int] = {
    "v": NM_RET_VOID,
    "i": NM_RET_I32,
    "u": NM_RET_U32,
    "p": NM_RET_PTR,
    "q": NM_RET_I64,
    "Q": NM_RET_U64,
    "f": NM_RET_F32,
    "d": NM_RET_F64,
}

_ARG_CHAR_MAP: Dict[str, int] = {
    "i": NM_ARG_I32,
    "u": NM_ARG_U32,
    "p": NM_ARG_PTR,
    "q": NM_ARG_I64,
    "Q": NM_ARG_U64,
    "f": NM_ARG_F32,
    "d": NM_ARG_F64,
}

# Module-level state for warn-once behavior
_WARNED_MISSING_CACHE = False
_WARNED_STALE_CACHE: Set[str] = set()


def _reset_warn_state() -> None:
    """Reset warning caches (primarily for unit test isolation)."""
    global _WARNED_MISSING_CACHE, _WARNED_STALE_CACHE
    _WARNED_MISSING_CACHE = False
    _WARNED_STALE_CACHE.clear()


# ---------------------------------------------------------------------------
# 2. Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedSig:
    """Parsed i386 function or data signature descriptor."""

    ret_kind: int = NM_RET_VOID
    nargs: int = 0
    is_variadic: bool = False
    abi_ext: int = 0
    arg_kinds: Tuple[int, ...] = ()
    data_size: Optional[int] = None
    sret_size: int = 0

    def encode_nm_fn_sig(self) -> bytes:
        """Pack into 20-byte struct nm_fn_sig_t (<BBBB16B)."""
        padded = list(self.arg_kinds[:NM_MAX_SIG_ARGS])
        while len(padded) < NM_MAX_SIG_ARGS:
            padded.append(NM_ARG_NONE)
        return struct.pack(
            "<BBBB16B",
            self.ret_kind,
            self.nargs,
            1 if self.is_variadic else 0,
            self.abi_ext,
            *padded,
        )


@dataclass
class LookupLibEntry:
    """Library-level record (lib <soname> <path_or_id> ...)."""

    soname: str
    path_or_id: Optional[str]
    tgt_soname: Optional[str]
    tgt_id: Optional[str]
    cls: str
    flags: int
    raw_flags: str
    notes: str
    line_no: int = 0


@dataclass
class LookupSymEntry:
    """Symbol-level record (sym <soname> <sym> ...)."""

    soname: str
    sym: str
    tgt_soname: Optional[str]
    tgt_sym: Optional[str]
    tgt_ver: Optional[str]
    cls: str
    sig_str: str
    parsed_sig: Optional[ParsedSig]
    flags: int
    raw_flags: str
    size: int
    reason: Optional[str] = None
    sym_ver: Optional[str] = None
    line_no: int = 0

    @property
    def is_data(self) -> bool:
        return bool(self.flags & NM_SYM_FLAG_DATA) or self.cls == "NATIVE_DATA"

    @property
    def is_variadic(self) -> bool:
        return bool(self.flags & NM_SYM_FLAG_VARIADIC) or (
            self.parsed_sig is not None and self.parsed_sig.is_variadic
        )

    @property
    def is_static_only(self) -> bool:
        return bool(self.flags & NM_SYM_FLAG_STATIC_ONLY)

    @property
    def is_runtime_only(self) -> bool:
        return bool(self.flags & NM_SYM_FLAG_RUNTIME_ONLY)

    @property
    def is_forbidden(self) -> bool:
        return self.cls in ("FORBIDDEN_ABI", "FORBIDDEN")

    @property
    def is_alias(self) -> bool:
        return self.cls == "ALIAS"

    @property
    def is_untyped(self) -> bool:
        return self.cls == "UNTYPED"


# ---------------------------------------------------------------------------
# 3. Signature DSL Parser
# ---------------------------------------------------------------------------

def parse_sig(sig_str: str) -> Optional[ParsedSig]:
    """Parse signature DSL string into a ParsedSig descriptor.

    Handles:
      - '-' or '' -> returns None
      - 'D<n>'    -> fixed data blob of size n
      - '<ret>:<args>[...]'
      - 'S<n>:<args>' where 1 <= n <= 16 (hidden struct return)

    Strictly rejects:
      - 'e' (Tier 1 rejected) -> raises ValueError
      - 'A<n>' (aggregate-by-value rejected) -> raises ValueError
      - more than 16 arguments -> raises ValueError
    """
    s = sig_str.strip()
    if not s or s == "-":
        return None

    # Strictly reject Tier-1 forbidden forms anywhere in signature string
    if "e" in s:
        raise ValueError(f"Rejected signature token 'e' in '{sig_str}'")
    if re.search(r"A\d+", s):
        raise ValueError(f"Rejected signature form 'A<n>' in '{sig_str}'")

    # Fixed data/blob argument: D<n>
    if s.startswith("D"):
        size_str = s[1:]
        if not size_str.isdigit():
            raise ValueError(f"Malformed data signature '{sig_str}'")
        data_sz = int(size_str)
        return ParsedSig(
            ret_kind=NM_RET_VOID,
            nargs=0,
            is_variadic=False,
            abi_ext=0,
            arg_kinds=(),
            data_size=data_sz,
            sret_size=0,
        )

    if ":" not in s:
        raise ValueError(f"Malformed function signature '{sig_str}' (missing ':')")

    ret_part, args_part = s.split(":", 1)

    # Return kind parsing
    abi_ext = 0
    sret_size = 0
    if ret_part.startswith("S"):
        # S<n> struct return prefix (1 <= n <= 16)
        sz_str = ret_part[1:]
        if not sz_str.isdigit():
            raise ValueError(f"Malformed S<n> return '{ret_part}' in '{sig_str}'")
        n = int(sz_str)
        if not (1 <= n <= 16):
            raise ValueError(f"S<n> size out of range [1, 16]: {n} in '{sig_str}'")
        sret_size = n
        abi_ext = nm_sig_ext_make_sret(n)
        ret_kind = NM_RET_PTR
    elif ret_part in _RET_CHAR_MAP:
        ret_kind = _RET_CHAR_MAP[ret_part]
    else:
        raise ValueError(f"Unknown return type '{ret_part}' in '{sig_str}'")

    # Variadic tail handling (...)
    is_variadic = False
    if args_part.endswith("..."):
        is_variadic = True
        args_part = args_part[:-3]
    elif "..." in args_part:
        # '...' embedded or with trailing tokens
        parts = args_part.split("...")
        if len(parts) == 2 and not parts[1]:
            is_variadic = True
            args_part = parts[0]
        else:
            raise ValueError(f"Invalid variadic placement in '{sig_str}'")

    # Argument kinds
    arg_kinds_list: List[int] = []
    for ch in args_part:
        if ch in _ARG_CHAR_MAP:
            arg_kinds_list.append(_ARG_CHAR_MAP[ch])
        else:
            raise ValueError(f"Unknown arg token '{ch}' in '{sig_str}'")

    if len(arg_kinds_list) > NM_MAX_SIG_ARGS:
        raise ValueError(
            f"Too many arguments ({len(arg_kinds_list)} > {NM_MAX_SIG_ARGS}) in '{sig_str}'"
        )

    return ParsedSig(
        ret_kind=ret_kind,
        nargs=len(arg_kinds_list),
        is_variadic=is_variadic,
        abi_ext=abi_ext,
        arg_kinds=tuple(arg_kinds_list),
        data_size=None,
        sret_size=sret_size,
    )


# ---------------------------------------------------------------------------
# 4. LookupTable Class
# ---------------------------------------------------------------------------

class LookupTable:
    """Parser, resolver, and validator for #!newminhh-lookup v1 tables."""

    def __init__(
        self,
        *,
        generator: Optional[str] = None,
        libc: Optional[str] = None,
        interp: Optional[str] = None,
        scan_roots: Optional[Dict[str, Optional[int]]] = None,
        rules_source: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> None:
        self.generator = generator
        self.libc = libc
        self.interp = interp
        self.scan_roots: Dict[str, Optional[int]] = scan_roots if scan_roots is not None else {}
        self.rules_source = rules_source
        self.source_path = source_path
        self.libs: Dict[str, LookupLibEntry] = {}
        self.syms: List[LookupSymEntry] = []
        # Key: (soname, sym, ver)
        self._sym_map: Dict[Tuple[str, str, Optional[str]], List[LookupSymEntry]] = {}

    @property
    def scans(self) -> List[Tuple[str, Optional[int]]]:
        """List of (scan_dir, mtime_ns) recorded in this table."""
        return list(self.scan_roots.items())

    def _add_sym_entry(self, entry: LookupSymEntry) -> None:
        self.syms.append(entry)
        key = (entry.soname, entry.sym, entry.sym_ver)
        self._sym_map.setdefault(key, []).append(entry)

    @classmethod
    def from_text(cls, text: str, source_path: str = "<memory>") -> "LookupTable":
        """Parse a full table from text string, strictly validating schema."""
        table = cls(source_path=source_path)
        lines = text.splitlines()
        header_seen = False

        for line_idx, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if not header_seen:
                if line != HEADER_MAGIC:
                    raise PatcherError(
                        f"{source_path}:{line_idx}: Missing or invalid magic header "
                        f"(expected '{HEADER_MAGIC}', found '{line}')"
                    )
                header_seen = True
                continue

            # Metadata directives
            if line.startswith("#@"):
                parts = line.split(maxsplit=1)
                directive = parts[0]
                val = parts[1].strip() if len(parts) > 1 else ""

                if directive == "#@generator":
                    table.generator = val
                elif directive == "#@libc":
                    table.libc = val
                elif directive == "#@interp":
                    table.interp = val
                elif directive == "#@rules":
                    table.rules_source = val
                elif directive == "#@scan":
                    # Exact v1 spec: tab-separated '#@scan\t<dir>\t<mtime_ns>' (3 columns)
                    # Also support legacy comma-separated, space-separated, :ns, and =ns
                    tab_parts = raw_line.strip().split("\t")
                    if len(tab_parts) >= 3 and tab_parts[0] == "#@scan":
                        p = tab_parts[1].strip()
                        ns_str = tab_parts[2].strip()
                        table.scan_roots[p] = int(ns_str) if ns_str.isdigit() else None
                    elif len(tab_parts) == 2 and tab_parts[0] == "#@scan" and not (":" in tab_parts[1] or "=" in tab_parts[1] or "," in tab_parts[1]):
                        table.scan_roots[tab_parts[1].strip()] = None
                    else:
                        for item in val.split(","):
                            item = item.strip()
                            if not item:
                                continue
                            if "\t" in item:
                                t_parts = item.split("\t", 1)
                                p = t_parts[0].strip()
                                ns_str = t_parts[1].strip()
                                table.scan_roots[p] = int(ns_str) if ns_str.isdigit() else None
                            elif ":" in item:
                                p, ns_str = item.rsplit(":", 1)
                                table.scan_roots[p] = int(ns_str) if ns_str.isdigit() else None
                            elif "=" in item:
                                p, ns_str = item.rsplit("=", 1)
                                table.scan_roots[p] = int(ns_str) if ns_str.isdigit() else None
                            elif " " in item:
                                s_parts = item.rsplit(" ", 1)
                                if s_parts[1].isdigit():
                                    table.scan_roots[s_parts[0].strip()] = int(s_parts[1])
                                else:
                                    table.scan_roots[item] = None
                            else:
                                table.scan_roots[item] = None
                else:
                    raise PatcherError(
                        f"{source_path}:{line_idx}: Unknown metadata directive '{directive}'"
                    )
                continue

            # Standard comments
            if line.startswith("#"):
                continue

            # Record rows must be tab-separated with 9 columns
            cols = raw_line.split("\t")
            if len(cols) != 9:
                raise PatcherError(
                    f"{source_path}:{line_idx}: Expected 9 tab-separated columns, found {len(cols)}"
                )

            rec_type = cols[0]
            if rec_type == "lib":
                soname = cols[1]
                path_or_id = None if cols[2] == "-" else cols[2]
                tgt_soname = None if cols[3] == "-" else cols[3]
                tgt_id = None if cols[4] == "-" else cols[4]
                class_str = cols[5]
                raw_flags = cols[6] if cols[6] != "-" else cols[7]
                notes = cols[8]

                flags = 0
                if raw_flags != "-":
                    for f in raw_flags.split(","):
                        flags |= FLAG_NAME_TO_MASK.get(f.strip(), 0)

                lib_entry = LookupLibEntry(
                    soname=soname,
                    path_or_id=path_or_id,
                    tgt_soname=tgt_soname,
                    tgt_id=tgt_id,
                    cls=class_str,
                    flags=flags,
                    raw_flags=raw_flags,
                    notes=notes,
                    line_no=line_idx,
                )
                table.libs[soname] = lib_entry

            elif rec_type == "sym":
                soname = cols[1]
                sym_raw = cols[2]
                tgt_soname = None if cols[3] == "-" else cols[3]
                tgt_sym_raw = None if cols[4] == "-" else cols[4]
                class_str = cols[5]
                sig_str = cols[6]
                raw_flags = cols[7]
                size_str = cols[8]

                # Parse sym and optional version on source symbol
                if "@" in sym_raw:
                    sym_name, sym_ver = sym_raw.split("@", 1)
                else:
                    sym_name, sym_ver = sym_raw, None

                # Parse target symbol and version
                if tgt_sym_raw and "@" in tgt_sym_raw:
                    tgt_sym, tgt_ver = tgt_sym_raw.split("@", 1)
                else:
                    tgt_sym, tgt_ver = tgt_sym_raw, None

                # Parse signature
                try:
                    parsed_sig = parse_sig(sig_str)
                except ValueError as exc:
                    raise PatcherError(
                        f"{source_path}:{line_idx}: Malformed signature '{sig_str}': {exc}"
                    ) from exc

                # Parse reason and flags
                reason: Optional[str] = None
                flags = 0
                if raw_flags != "-":
                    rem = raw_flags
                    idx = rem.find("REASON=")
                    if idx != -1:
                        reason_part = rem[idx + 7:]
                        # Reason extends until comma before a known flag or end of string
                        for fn in FLAG_NAME_TO_MASK:
                            pos = reason_part.find("," + fn)
                            if pos != -1:
                                flags |= FLAG_NAME_TO_MASK[fn]
                                reason_part = reason_part[:pos]
                        reason = reason_part
                        rem = rem[:idx].rstrip(",")

                    for token in rem.split(","):
                        token = token.strip()
                        if token in FLAG_NAME_TO_MASK:
                            flags |= FLAG_NAME_TO_MASK[token]

                # Also inspect size_str for reason or size
                size = 0
                if size_str != "-":
                    if size_str.startswith("REASON="):
                        if reason is None:
                            reason = size_str[7:]
                    elif size_str.isdigit():
                        size = int(size_str)

                if class_str == "NATIVE_DATA":
                    flags |= NM_SYM_FLAG_DATA
                    if size == 0 and parsed_sig and parsed_sig.data_size is not None:
                        size = parsed_sig.data_size
                elif class_str == "PATCHED_CDECL":
                    flags |= NM_SYM_FLAG_PATCHED

                if parsed_sig and parsed_sig.is_variadic:
                    flags |= NM_SYM_FLAG_VARIADIC

                sym_entry = LookupSymEntry(
                    soname=soname,
                    sym=sym_name,
                    tgt_soname=tgt_soname,
                    tgt_sym=tgt_sym,
                    tgt_ver=tgt_ver,
                    cls=class_str,
                    sig_str=sig_str,
                    parsed_sig=parsed_sig,
                    flags=flags,
                    raw_flags=raw_flags,
                    size=size,
                    reason=reason,
                    sym_ver=sym_ver,
                    line_no=line_idx,
                )
                table._add_sym_entry(sym_entry)

            else:
                raise PatcherError(
                    f"{source_path}:{line_idx}: Unknown record type '{rec_type}' (expected 'lib' or 'sym')"
                )

        if not header_seen:
            raise PatcherError(f"{source_path}: Empty lookup file without header")

        return table

    @classmethod
    def from_file(cls, path: str) -> "LookupTable":
        """Load and parse a table from a specific file path."""
        if not os.path.exists(path):
            raise PatcherError(f"Lookup table file not found: '{path}'")
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            raise PatcherError(f"Failed to read lookup table '{path}': {exc}") from exc
        return cls.from_text(text, source_path=path)

    @classmethod
    def load(cls, path: Optional[str] = None, *, warn_once: bool = True) -> "LookupTable":
        """Load lookup table with priority resolution, fallback, and staleness check.

        1. Explicit path: MUST exist or raises PatcherError.
        2. NEWMINHH_LOOKUP_TABLE env var: MUST exist or raises PatcherError.
        3. /var/cache/newminhh/lookup_table.txt
        4. Fallbacks if missing/unreadable:
           a) from ._lookup_registry_gen import EMBEDDED_LOOKUP_BASE_TEXT
           b) /usr/share/newminhh/lookup_base.txt
           c) repo-relative ../../../lookup/rules.d/ (00-libc.txt or merged rules.d/*.txt)
           Logs ONE warning to sys.stderr on fallback.
        5. Checks is_stale() and logs ONE warning to sys.stderr if stale (never writes cache!).
        """
        global _WARNED_MISSING_CACHE, _WARNED_STALE_CACHE

        table: Optional[LookupTable] = None

        if path is not None:
            if not os.path.exists(path):
                raise PatcherError(f"Explicit lookup table not found: '{path}'")
            table = cls.from_file(path)

        if table is None:
            env_path = os.environ.get("NEWMINHH_LOOKUP_TABLE")
            if env_path:
                if not os.path.exists(env_path):
                    raise PatcherError(f"NEWMINHH_LOOKUP_TABLE file not found: '{env_path}'")
                table = cls.from_file(env_path)

        if table is None and os.path.exists(DEFAULT_CACHE_PATH):
            try:
                table = cls.from_file(DEFAULT_CACHE_PATH)
            except Exception as exc:
                if not _WARNED_MISSING_CACHE or not warn_once:
                    sys.stderr.write(
                        f"[newminhh:lookup] WARNING: Cache '{DEFAULT_CACHE_PATH}' invalid ({exc}); "
                        f"falling back to base table\n"
                    )
                    sys.stderr.flush()
                    _WARNED_MISSING_CACHE = True

        if table is None:
            # Fallback chain
            if not _WARNED_MISSING_CACHE or not warn_once:
                sys.stderr.write(
                    f"[newminhh:lookup] WARNING: Lookup cache missing; using static fallback rules\n"
                )
                sys.stderr.flush()
                _WARNED_MISSING_CACHE = True

            # a) Try embedded base text
            try:
                from ._lookup_registry_gen import EMBEDDED_LOOKUP_BASE_TEXT  # type: ignore[import-not-found]
                table = cls.from_text(EMBEDDED_LOOKUP_BASE_TEXT, source_path="<embedded>")
            except (ImportError, AttributeError):
                pass

            # b) Try /usr/share/newminhh/lookup_base.txt
            if table is None and os.path.exists(DEFAULT_BASE_PATH):
                try:
                    table = cls.from_file(DEFAULT_BASE_PATH)
                except Exception:
                    pass

            # c) Try repo-relative rules
            if table is None:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                candidate_paths = [
                    os.path.abspath(os.path.join(base_dir, "..", "..", "..", "lookup", "rules.d", "00-libc.txt")),
                    os.path.abspath(os.path.join(base_dir, "..", "..", "lookup", "rules.d", "00-libc.txt")),
                    os.path.abspath(os.path.join(base_dir, "..", "..", "..", "src", "lookup", "rules.d", "00-libc.txt")),
                ]
                for c_path in candidate_paths:
                    if os.path.exists(c_path):
                        table = cls.from_file(c_path)
                        break

            # If 00-libc.txt not found, try merged repo-relative rules.d directory
            if table is None:
                rules_dirs = [
                    os.path.abspath(os.path.join(base_dir, "..", "..", "..", "lookup", "rules.d")),
                    os.path.abspath(os.path.join(base_dir, "..", "..", "lookup", "rules.d")),
                    os.path.abspath(os.path.join(base_dir, "..", "..", "..", "src", "lookup", "rules.d")),
                ]
                for r_dir in rules_dirs:
                    if os.path.isdir(r_dir):
                        merged_lines = [HEADER_MAGIC, f"#@rules {r_dir}"]
                        for fname in sorted(os.listdir(r_dir)):
                            if fname.endswith(".txt"):
                                fpath = os.path.join(r_dir, fname)
                                try:
                                    with open(fpath, "r", encoding="utf-8") as rf:
                                        for r_line in rf:
                                            r_strip = r_line.strip()
                                            if r_strip and not r_strip.startswith("#"):
                                                merged_lines.append(r_strip)
                                except OSError:
                                    continue
                        if len(merged_lines) > 2:
                            table = cls.from_text("\n".join(merged_lines) + "\n", source_path=r_dir)
                            break

        if table is None:
            raise PatcherError(
                "No lookup table could be loaded (cache, embedded base, and fallback rules all missing)"
            )

        # Staleness check
        if table.is_stale():
            cache_id = table.source_path or "<table source>"
            if cache_id not in _WARNED_STALE_CACHE or not warn_once:
                sys.stderr.write(
                    f"[newminhh:lookup] WARNING: Lookup table '{cache_id}' is stale; "
                    f"underlying scan directories have changed\n"
                )
                sys.stderr.flush()
                _WARNED_STALE_CACHE.add(cache_id)

        return table

    def is_stale(self) -> bool:
        """Check if any recorded scan roots have changed mtime."""
        if not self.scan_roots:
            return False

        for scan_dir, recorded_ns in self.scan_roots.items():
            if recorded_ns is not None:
                try:
                    st = os.stat(scan_dir)
                    if st.st_mtime_ns != recorded_ns:
                        return True
                except OSError:
                    return True
            else:
                if self.source_path and os.path.exists(self.source_path):
                    try:
                        cache_ns = os.stat(self.source_path).st_mtime_ns
                        dir_ns = os.stat(scan_dir).st_mtime_ns
                        if dir_ns > cache_ns:
                            return True
                    except OSError:
                        return True
        return False

    def resolve_lib(self, soname: str) -> Optional[LookupLibEntry]:
        """Resolve library entry by soname, falling back to '*'."""
        return self.libs.get(soname) or self.libs.get("*")

    def resolve_sym(
        self,
        soname: str,
        sym: str,
        ver: Optional[str] = None,
        *,
        mode: str = "static",
        _depth: int = 0,
    ) -> Optional[LookupSymEntry]:
        """Resolve symbol through (soname,sym,ver) -> (soname,sym,None) -> (*,sym,ver) -> (*,sym,None).

        Filters STATIC_ONLY and RUNTIME_ONLY based on mode ("static" or "runtime").
        Follows ALIAS chains up to max depth 4.
        """
        if _depth > 4:
            raise PatcherError(
                f"Alias cycle or exceeded max alias depth (4) resolving symbol '{soname}:{sym}'"
            )

        search_keys: List[Optional[Tuple[str, str, Optional[str]]]] = [
            (soname, sym, ver) if ver is not None else None,
            (soname, sym, None),
            ("*", sym, ver) if ver is not None else None,
            ("*", sym, None),
        ]

        candidate: Optional[LookupSymEntry] = None
        for key in search_keys:
            if key is None:
                continue
            entries = self._sym_map.get(key)
            if not entries:
                continue
            for entry in entries:
                if mode == "static" and (entry.flags & NM_SYM_FLAG_RUNTIME_ONLY):
                    continue
                if mode == "runtime" and (entry.flags & NM_SYM_FLAG_STATIC_ONLY):
                    continue
                candidate = entry
                break
            if candidate is not None:
                break

        # Fallback when ver is None: check any entry registered for (soname, sym)
        if candidate is None and ver is None:
            for s_soname in (soname, "*"):
                for (k_so, k_sym, _), entries in self._sym_map.items():
                    if k_so == s_soname and k_sym == sym:
                        for entry in entries:
                            if mode == "static" and (entry.flags & NM_SYM_FLAG_RUNTIME_ONLY):
                                continue
                            if mode == "runtime" and (entry.flags & NM_SYM_FLAG_STATIC_ONLY):
                                continue
                            candidate = entry
                            break
                        if candidate is not None:
                            break
                if candidate is not None:
                    break

        if candidate is None:
            return None

        # Follow ALIAS chain if necessary
        if candidate.cls == "ALIAS":
            tgt_soname = candidate.tgt_soname or soname
            tgt_sym = candidate.tgt_sym
            tgt_ver = candidate.tgt_ver or ver
            if not tgt_sym:
                raise PatcherError(
                    f"Malformed ALIAS for '{candidate.soname}:{candidate.sym}' (missing target symbol)"
                )
            target = self.resolve_sym(tgt_soname, tgt_sym, tgt_ver, mode=mode, _depth=_depth + 1)
            if target is None:
                raise PatcherError(
                    f"Unresolvable ALIAS '{candidate.soname}:{candidate.sym}' -> '{tgt_soname}:{tgt_sym}'"
                )
            return target

        return candidate

    def check_bindable(
        self,
        entry: Optional[LookupSymEntry],
        *,
        sym: str,
        soname: str,
        allow_untyped_unsafe: bool = False,
    ) -> LookupSymEntry:
        """Check symbol bindability according to newminhh ABI rules.

        - Fails loudly with PatcherError on UNTYPED or missing entry,
          unless allow_untyped_unsafe=True, which warns and returns synthetic 6x NM_ARG_U32.
        - ALWAYS fails loudly on FORBIDDEN_ABI with reason.
        """
        return check_bindable(
            entry, sym=sym, soname=soname, allow_untyped_unsafe=allow_untyped_unsafe
        )


# ---------------------------------------------------------------------------
# 5. Bindability Helper
# ---------------------------------------------------------------------------

def check_bindable(
    entry: Optional[LookupSymEntry],
    *,
    sym: str,
    soname: str,
    allow_untyped_unsafe: bool = False,
) -> LookupSymEntry:
    """Check symbol bindability according to newminhh ABI rules.

    - Fails loudly with PatcherError on UNTYPED or missing entry,
      unless allow_untyped_unsafe=True (warns and returns synthetic 6x NM_ARG_U32).
    - ALWAYS fails loudly on FORBIDDEN_ABI (with reason).
    """
    if entry is not None and entry.cls in ("FORBIDDEN_ABI", "FORBIDDEN"):
        reason_str = f" ({entry.reason})" if entry.reason else ""
        raise PatcherError(f"Symbol '{soname}:{sym}' has FORBIDDEN_ABI{reason_str}")

    if entry is None or entry.cls == "UNTYPED":
        if allow_untyped_unsafe:
            sys.stderr.write(
                f"[newminhh:lookup] WARNING: Untyped symbol '{soname}:{sym}' permitted "
                f"under allow_untyped_unsafe; using synthetic 6x U32 signature\n"
            )
            sys.stderr.flush()
            synth_sig = ParsedSig(
                ret_kind=NM_RET_U32,
                nargs=6,
                is_variadic=False,
                abi_ext=0,
                arg_kinds=(
                    NM_ARG_U32,
                    NM_ARG_U32,
                    NM_ARG_U32,
                    NM_ARG_U32,
                    NM_ARG_U32,
                    NM_ARG_U32,
                ),
            )
            return LookupSymEntry(
                soname=soname,
                sym=sym,
                tgt_soname=soname,
                tgt_sym=sym,
                tgt_ver=None,
                cls="NATIVE_TYPED",
                sig_str="u:uuuuuu",
                parsed_sig=synth_sig,
                flags=NM_SYM_FLAG_NATIVE,
                raw_flags="-",
                size=0,
                reason=None,
                sym_ver=None,
                line_no=0,
            )
        raise PatcherError(
            f"Symbol '{soname}:{sym}' is untyped or missing from lookup table"
        )

    return entry


def load_lookup_table(path: Optional[str] = None, *, warn_once: bool = True) -> LookupTable:
    """Convenience functional loader for LookupTable."""
    return LookupTable.load(path=path, warn_once=warn_once)


# ---------------------------------------------------------------------------
# 6. Inline Unit Tests (Executed when invoked directly)
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    """Run comprehensive unit tests verifying all lookup specifications."""
    import tempfile

    _reset_warn_state()

    # 1. Constants
    assert NM_RET_VOID == 0 and NM_RET_F64 == 7
    assert NM_ARG_NONE == 0 and NM_ARG_F64 == 7
    assert NM_MAX_SIG_ARGS == 16
    assert NM_SIG_EXT_SRET == 1 and NM_SIG_EXT_CALLEE_POP4 == 2
    assert nm_sig_ext_make_sret(8) == (1 | 2 | (8 << 3))
    assert nm_sig_ext_sret_size(nm_sig_ext_make_sret(8)) == 8
    assert nm_sig_ext_sret_size(nm_sig_ext_make_sret(16)) == 16

    # 2. Signature parsing & encoding
    assert parse_sig("-") is None
    assert parse_sig("") is None

    # D<n>
    dsig = parse_sig("D4")
    assert dsig is not None and dsig.data_size == 4 and dsig.ret_kind == NM_RET_VOID

    # v:pp
    sig1 = parse_sig("v:pp")
    assert sig1 is not None
    assert sig1.ret_kind == NM_RET_VOID
    assert sig1.nargs == 2
    assert sig1.arg_kinds == (NM_ARG_PTR, NM_ARG_PTR)
    assert not sig1.is_variadic
    enc1 = sig1.encode_nm_fn_sig()
    assert len(enc1) == 20
    assert enc1[:4] == struct.pack("<BBBB", NM_RET_VOID, 2, 0, 0)
    assert enc1[4:6] == bytes([NM_ARG_PTR, NM_ARG_PTR])
    assert enc1[6:] == b"\x00" * 14

    # u:
    sig2 = parse_sig("u:")
    assert sig2 is not None and sig2.ret_kind == NM_RET_U32 and sig2.nargs == 0

    # i:p...
    sig3 = parse_sig("i:p...")
    assert sig3 is not None and sig3.ret_kind == NM_RET_I32 and sig3.is_variadic and sig3.nargs == 1
    assert sig3.encode_nm_fn_sig()[2] == 1

    # S8:ii
    sig4 = parse_sig("S8:ii")
    assert sig4 is not None
    assert sig4.ret_kind == NM_RET_PTR
    assert sig4.sret_size == 8
    assert sig4.abi_ext == nm_sig_ext_make_sret(8)
    assert sig4.nargs == 2
    assert sig4.arg_kinds == (NM_ARG_I32, NM_ARG_I32)

    # Rejection of 'e', 'A<n>', >16 args, S0, S17
    for rejected in ("e:i", "v:e", "v:A4", "A8:i", "S0:i", "S17:i", "v:" + "i" * 17):
        try:
            parse_sig(rejected)
            assert False, f"Expected ValueError for {rejected}"
        except ValueError:
            pass

    # 3. LookupTable.from_text validation
    valid_text = (
        "#!newminhh-lookup v1\n"
        "#@generator test-gen 1.0\n"
        "#@libc glibc 2.38\n"
        "#@interp /libx32/ld-linux-x32.so.2\n"
        "#@rules test-rules\n"
        "lib\tlibc.so.6\t-\tlibc.so.6\t-\tNATIVE\t-\t-\t-\n"
        "sym\tlibc.so.6\tread\tlibc.so.6\tread@GLIBC_2.0\tNATIVE_TYPED\ti:ipu\t-\t0\n"
        "sym\tlibc.so.6\tstat\t-\t-\tFORBIDDEN_ABI\t-\tREASON=struct-stat-i386\t0\n"
        "sym\tlibc.so.6\tstdin\tlibc.so.6\tstdin\tNATIVE_DATA\tD4\tDATA\t4\n"
        "sym\tlibc.so.6\t__read\tlibc.so.6\tread\tALIAS\t-\t-\t0\n"
        "sym\t*\toptind\tlibc.so.6\toptind\tNATIVE_DATA\tD4\tDATA\t4\n"
    )
    t = LookupTable.from_text(valid_text, source_path="test_table.txt")
    assert t.generator == "test-gen 1.0"
    assert t.libc == "glibc 2.38"
    assert t.interp == "/libx32/ld-linux-x32.so.2"
    assert t.rules_source == "test-rules"
    assert "libc.so.6" in t.libs

    # Missing header check
    try:
        LookupTable.from_text("lib\tfoo\t-\tfoo\t-\tNATIVE\t-\t-\t-\n")
        assert False, "Expected PatcherError on missing header"
    except PatcherError as err:
        assert "Missing or invalid magic header" in str(err)

    # Malformed column count check
    try:
        LookupTable.from_text("#!newminhh-lookup v1\nlib\tlibc.so.6\t-\tNATIVE\n")
        assert False, "Expected PatcherError on bad column count"
    except PatcherError as err:
        assert "Expected 9 tab-separated columns" in str(err)

    # 4. Symbol Resolution & ALIAS
    read_sym = t.resolve_sym("libc.so.6", "read")
    assert read_sym is not None and read_sym.sym == "read" and read_sym.cls == "NATIVE_TYPED"
    assert read_sym.tgt_ver == "GLIBC_2.0"

    # Alias chain (__read -> read)
    alias_sym = t.resolve_sym("libc.so.6", "__read")
    assert alias_sym is not None and alias_sym.sym == "read"

    # Wildcard soname resolution
    opt_sym = t.resolve_sym("some_other.so", "optind")
    assert opt_sym is not None and opt_sym.sym == "optind" and opt_sym.is_data

    # Library resolution
    assert t.resolve_lib("libc.so.6") is not None
    assert t.resolve_lib("missing.so") is None

    # Alias cycle detection
    cycle_text = (
        "#!newminhh-lookup v1\n"
        "sym\tlib.so\ta\tlib.so\tb\tALIAS\t-\t-\t0\n"
        "sym\tlib.so\tb\tlib.so\tc\tALIAS\t-\t-\t0\n"
        "sym\tlib.so\tc\tlib.so\td\tALIAS\t-\t-\t0\n"
        "sym\tlib.so\td\tlib.so\te\tALIAS\t-\t-\t0\n"
        "sym\tlib.so\te\tlib.so\ta\tALIAS\t-\t-\t0\n"
    )
    t_cycle = LookupTable.from_text(cycle_text)
    try:
        t_cycle.resolve_sym("lib.so", "a")
        assert False, "Expected PatcherError on alias cycle"
    except PatcherError as err:
        assert "Alias cycle" in str(err)

    # 5. check_bindable tests
    bindable = t.check_bindable(read_sym, sym="read", soname="libc.so.6")
    assert bindable is read_sym

    # FORBIDDEN_ABI always fails loudly
    stat_sym = t.resolve_sym("libc.so.6", "stat")
    assert stat_sym is not None and stat_sym.is_forbidden
    try:
        t.check_bindable(stat_sym, sym="stat", soname="libc.so.6", allow_untyped_unsafe=True)
        assert False, "Expected PatcherError on FORBIDDEN_ABI"
    except PatcherError as err:
        assert "FORBIDDEN_ABI" in str(err)
        assert "struct-stat-i386" in str(err)

    # Missing / untyped symbol
    try:
        t.check_bindable(None, sym="unknown_fn", soname="libfoo.so", allow_untyped_unsafe=False)
        assert False, "Expected PatcherError on untyped symbol"
    except PatcherError as err:
        assert "untyped or missing" in str(err)

    # Missing with allow_untyped_unsafe=True
    synth = t.check_bindable(None, sym="unknown_fn", soname="libfoo.so", allow_untyped_unsafe=True)
    assert synth.cls == "NATIVE_TYPED"
    assert synth.parsed_sig is not None and synth.parsed_sig.nargs == 6
    assert synth.parsed_sig.ret_kind == NM_RET_U32
    assert synth.parsed_sig.arg_kinds == (NM_ARG_U32,) * 6

    # 6. Staleness check
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_ns = os.stat(tmp_dir).st_mtime_ns
        stale_text = (
            "#!newminhh-lookup v1\n"
            f"#@scan {tmp_dir}:{dir_ns}\n"
            "sym\tlib.so\tfn\tlib.so\tfn\tNATIVE_TYPED\tv:\t-\t0\n"
        )
        t_fresh = LookupTable.from_text(stale_text)
        assert not t_fresh.is_stale()

        stale_text_mismatch = (
            "#!newminhh-lookup v1\n"
            f"#@scan {tmp_dir}:{dir_ns + 9999}\n"
            "sym\tlib.so\tfn\tlib.so\tfn\tNATIVE_TYPED\tv:\t-\t0\n"
        )
        t_stale = LookupTable.from_text(stale_text_mismatch)
        assert t_stale.is_stale()

    # 7. File loading & fallback
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf_name = tf.name
        tf.write(valid_text)

    try:
        loaded_tf = LookupTable.load(tf_name)
        assert loaded_tf.resolve_sym("libc.so.6", "read") is not None
    finally:
        if os.path.exists(tf_name):
            os.remove(tf_name)

    # Non-existent explicit path
    try:
        LookupTable.load("/non/existent/path/lookup.txt")
        assert False, "Expected PatcherError on missing explicit path"
    except PatcherError:
        pass


if __name__ == "__main__":
    _run_tests()
    print("All lookup.py unit tests passed successfully.")
