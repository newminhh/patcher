"""Stage 0 — Strict ELF32 Validation & CLI Orchestration for newminhh-patch."""

from __future__ import annotations

import argparse
import os
import struct
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

from . import PatcherError

EI_MAG = b"\x7fELF"
ELFCLASS32 = 1
ELFDATA2LSB = 1
EV_CURRENT = 1
ET_EXEC = 2
ET_DYN = 3
EM_386 = 3

PT_NULL = 0
PT_LOAD = 1
PT_DYNAMIC = 2
PT_INTERP = 3
PT_NOTE = 4
PT_PHDR = 6
PT_TLS = 7

SHT_NULL = 0
SHT_NOBITS = 8

DT_NULL = 0
DT_PLTRELSZ = 2
DT_STRTAB = 5
DT_SYMTAB = 6
DT_RELA = 7
DT_RELASZ = 8
DT_STRSZ = 10
DT_SYMENT = 11
DT_INIT = 12
DT_FINI = 13
DT_REL = 17
DT_RELSZ = 18
DT_JMPREL = 23
DT_INIT_ARRAY = 25
DT_FINI_ARRAY = 26
DT_INIT_ARRAYSZ = 27
DT_FINI_ARRAYSZ = 28


def _log_error(msg: str) -> None:
    sys.stderr.write(f"[newminhh:patcher] ERROR: {msg}\n")
    sys.stderr.flush()


def _vaddr_to_offset(loads: List[Tuple[int, int, int, int]], vaddr: int, size: int = 1) -> Optional[int]:
    for p_vaddr, p_offset, p_filesz, p_memsz in loads:
        if p_filesz == 0:
            continue
        if p_vaddr <= vaddr and (vaddr + size) <= (p_vaddr + p_filesz):
            return p_offset + (vaddr - p_vaddr)
    return None


def validate_elf32_i386(data: bytes, path: str = "<input>") -> None:
    """Strictly validate an ELF32 little-endian i386 executable or shared object."""
    fsize = len(data)
    if fsize < 52:
        raise PatcherError(
            f"'{path}': truncated ELF header ({fsize} bytes < 52 required for Elf32_Ehdr)"
        )

    if data[:4] != EI_MAG:
        raise PatcherError(
            f"'{path}': invalid ELF magic {data[:4]!r} (expected b'\\x7fELF')"
        )

    ei_class = data[4]
    if ei_class != ELFCLASS32:
        raise PatcherError(
            f"'{path}': unsupported ELF class EI_CLASS={ei_class} (expected ELFCLASS32=1)"
        )

    ei_data = data[5]
    if ei_data != ELFDATA2LSB:
        raise PatcherError(
            f"'{path}': unsupported endianness EI_DATA={ei_data} (expected ELFDATA2LSB=1)"
        )

    ei_version = data[6]
    if ei_version != EV_CURRENT:
        raise PatcherError(
            f"'{path}': invalid EI_VERSION={ei_version} (expected 1)"
        )

    (
        e_type,
        e_machine,
        e_version,
        e_entry,
        e_phoff,
        e_shoff,
        _e_flags,
        e_ehsize,
        e_phentsize,
        e_phnum,
        e_shentsize,
        e_shnum,
        e_shstrndx,
    ) = struct.unpack_from("<HHIIIIIHHHHHH", data, 16)

    if e_machine != EM_386:
        raise PatcherError(
            f"'{path}': unsupported e_machine={e_machine} (expected EM_386=3)"
        )

    if e_type not in (ET_EXEC, ET_DYN):
        raise PatcherError(
            f"'{path}': unsupported ELF e_type={e_type} (expected ET_EXEC=2 or ET_DYN=3)"
        )

    if e_version != EV_CURRENT or e_ehsize < 52:
        raise PatcherError(
            f"'{path}': invalid ELF version/ehsize (e_version={e_version}, e_ehsize={e_ehsize})"
        )

    if e_phnum == 0 or e_phentsize != 32:
        raise PatcherError(
            f"'{path}': invalid program header table (e_phnum={e_phnum}, e_phentsize={e_phentsize})"
        )

    ph_end = e_phoff + e_phnum * e_phentsize
    if e_phoff < 52 or ph_end > fsize:
        raise PatcherError(
            f"'{path}': program header table out of bounds (phoff=0x{e_phoff:x}, end=0x{ph_end:x}, filesize=0x{fsize:x})"
        )

    loads: List[Tuple[int, int, int, int]] = []
    rx_ranges: List[Tuple[int, int]] = []
    dyn_phdr: Optional[Tuple[int, int, int]] = None

    for idx in range(e_phnum):
        off = e_phoff + idx * e_phentsize
        p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_flags, p_align = struct.unpack_from(
            "<IIIIIIII", data, off
        )

        if p_filesz > p_memsz and p_type == PT_LOAD:
            raise PatcherError(
                f"'{path}': phdr[{idx}] PT_LOAD has p_filesz (0x{p_filesz:x}) > p_memsz (0x{p_memsz:x})"
            )

        if p_type in (PT_LOAD, PT_DYNAMIC, PT_INTERP, PT_PHDR, PT_NOTE, PT_TLS) and p_filesz > 0:
            if p_offset + p_filesz > fsize:
                raise PatcherError(
                    f"'{path}': phdr[{idx}] (type={p_type}) file range [0x{p_offset:x}, 0x{p_offset + p_filesz:x}) exceeds EOF 0x{fsize:x}"
                )

        if p_type == PT_LOAD:
            if p_vaddr + p_memsz > 0x100000000:
                raise PatcherError(
                    f"'{path}': phdr[{idx}] PT_LOAD overflows 32-bit address space (0x{p_vaddr:x} + 0x{p_memsz:x})"
                )
            loads.append((p_vaddr, p_offset, p_filesz, p_memsz))
            if p_flags & 1:  # PF_X
                rx_ranges.append((p_vaddr, p_vaddr + p_memsz))
        elif p_type == PT_DYNAMIC:
            dyn_phdr = (p_offset, p_vaddr, p_filesz)

    if not loads:
        raise PatcherError(f"'{path}': no PT_LOAD segments present")

    if e_entry != 0:
        if not any(lo <= e_entry < hi for lo, hi in rx_ranges):
            raise PatcherError(
                f"'{path}': e_entry=0x{e_entry:08x} is not inside any executable PT_LOAD segment"
            )

    if dyn_phdr is not None:
        d_off, d_va, d_sz = dyn_phdr
        if d_sz < 8 or d_off + d_sz > fsize:
            raise PatcherError(
                f"'{path}': PT_DYNAMIC out of bounds (off=0x{d_off:x}, sz=0x{d_sz:x})"
            )
        if _vaddr_to_offset(loads, d_va, min(d_sz, 8)) is None:
            raise PatcherError(
                f"'{path}': PT_DYNAMIC vaddr=0x{d_va:08x} is not mapped by any PT_LOAD"
            )

        tags: Dict[int, int] = {}
        found_null = False
        for pos in range(d_off, d_off + d_sz - 7, 8):
            d_tag, d_val = struct.unpack_from("<iI", data, pos)
            if d_tag == DT_NULL:
                found_null = True
                break
            tags[d_tag] = d_val

        if not found_null:
            raise PatcherError(f"'{path}': PT_DYNAMIC missing DT_NULL terminator")

        strsz = tags.get(DT_STRSZ, 1)
        if DT_STRTAB in tags:
            if _vaddr_to_offset(loads, tags[DT_STRTAB], max(1, strsz)) is None:
                raise PatcherError(
                    f"'{path}': corrupt DT_STRTAB=0x{tags[DT_STRTAB]:08x} (strsz=0x{strsz:x}) outside PT_LOAD"
                )
        if DT_SYMTAB in tags:
            syment = tags.get(DT_SYMENT, 16)
            if syment != 16:
                raise PatcherError(f"'{path}': invalid DT_SYMENT={syment} (expected 16)")
            if _vaddr_to_offset(loads, tags[DT_SYMTAB], 16) is None:
                raise PatcherError(
                    f"'{path}': corrupt DT_SYMTAB=0x{tags[DT_SYMTAB]:08x} outside PT_LOAD"
                )
        for ptr_tag, sz_tag, name in (
            (DT_REL, DT_RELSZ, "DT_REL"),
            (DT_RELA, DT_RELASZ, "DT_RELA"),
            (DT_JMPREL, DT_PLTRELSZ, "DT_JMPREL"),
            (DT_INIT_ARRAY, DT_INIT_ARRAYSZ, "DT_INIT_ARRAY"),
            (DT_FINI_ARRAY, DT_FINI_ARRAYSZ, "DT_FINI_ARRAY"),
        ):
            if ptr_tag in tags:
                sz_val = tags.get(sz_tag, 0)
                if sz_val > 0 and _vaddr_to_offset(loads, tags[ptr_tag], sz_val) is None:
                    raise PatcherError(
                        f"'{path}': corrupt {name}=0x{tags[ptr_tag]:08x} (sz=0x{sz_val:x}) outside PT_LOAD"
                    )

    if e_shoff != 0:
        if e_shentsize != 40 or e_shoff + e_shnum * e_shentsize > fsize:
            raise PatcherError(
                f"'{path}': section header table out of bounds (shoff=0x{e_shoff:x}, shnum={e_shnum})"
            )
        if e_shnum > 0 and e_shstrndx != 0xFFFF and e_shstrndx >= e_shnum:
            raise PatcherError(
                f"'{path}': e_shstrndx={e_shstrndx} >= e_shnum={e_shnum}"
            )
        for idx in range(e_shnum):
            soff = e_shoff + idx * e_shentsize
            _sh_name, sh_type, _sh_flags, _sh_addr, sh_offset, sh_size = struct.unpack_from(
                "<IIIIII", data, soff
            )
            if sh_type != SHT_NULL and sh_type != SHT_NOBITS and sh_size > 0:
                if sh_offset + sh_size > fsize:
                    raise PatcherError(
                        f"'{path}': section[{idx}] range [0x{sh_offset:x}, 0x{sh_offset + sh_size:x}) exceeds EOF 0x{fsize:x}"
                    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="newminhh-patch",
        description="Static i386 -> Long Mode x32 ELF patcher backed by re-env",
    )
    parser.add_argument("input_elf", help="Input i386 ELF32 executable or shared object")
    parser.add_argument("-o", "--output", required=True, help="Output x32 EM_X86_64 ELF32 path")
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--hybrid-fallback",
        action="store_true",
        help="Force hybrid fallback mode (embed .newminhh.orig + link libnewminhh_dbt.so)",
    )
    mode_grp.add_argument(
        "--no-hybrid-fallback",
        action="store_true",
        help="Force pure static mode without libnewminhh_dbt.so (fail if any function requires DBT)",
    )
    parser.add_argument(
        "--use-ghidra",
        action="store_true",
        help="Invoke /opt/ghidra/support/analyzeHeadless to corroborate CFG/switch recovery",
    )
    parser.add_argument(
        "--use-r2",
        action="store_true",
        help="Invoke radare2 (r2) to corroborate CFG/switch recovery (auto-enabled on stripped ELFs)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose diagnostics")

    args = parser.parse_args(argv)

    if args.no_hybrid_fallback:
        hybrid_mode = "static"
    elif args.hybrid_fallback:
        hybrid_mode = "hybrid"
    else:
        hybrid_mode = "auto"

    try:
        if not os.path.isfile(args.input_elf):
            raise PatcherError(f"input file not found: '{args.input_elf}'")

        with open(args.input_elf, "rb") as f:
            raw_bytes = f.read()

        # Stage 0: Strict validation
        validate_elf32_i386(raw_bytes, args.input_elf)

        # Late imports so Stage 0 validation stays instant and self-contained on N01-N06
        from .analyzer import analyze_binary
        from .rewriter import rewrite_binary
        from .elf_builder import build_x32_elf

        # Stage 1: Static RE analysis
        analysis = analyze_binary(
            args.input_elf,
            raw_bytes,
            use_r2=args.use_r2,
            use_ghidra=args.use_ghidra,
            verbose=args.verbose,
        )

        # Check static-only vs hybrid mode constraints
        fallback_funcs = [
            va for va, cls in analysis.func_classifications.items() if cls == "REQUIRES_DBT_FALLBACK"
        ]
        if hybrid_mode == "static" and fallback_funcs:
            raise PatcherError(
                f"'{args.input_elf}': --no-hybrid-fallback specified but {len(fallback_funcs)} "
                f"function(s) require DBT fallback: {[hex(x) for x in fallback_funcs[:5]]}"
            )

        # Stage 2: Rewriting & ELF building
        rewritten = rewrite_binary(analysis, hybrid_mode=hybrid_mode, verbose=args.verbose)
        out_bytes = build_x32_elf(analysis, rewritten, hybrid_mode=hybrid_mode, verbose=args.verbose)

        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".newminhh_tmp_", dir=out_dir)
        try:
            with os.fdopen(fd, "wb") as tf:
                tf.write(out_bytes)
            os.chmod(tmp_path, 0o755)
            os.replace(tmp_path, args.output)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        if args.verbose:
            sys.stderr.write(
                f"[newminhh:patcher] OK: '{args.input_elf}' -> '{args.output}' "
                f"(mode={hybrid_mode}, funcs={len(analysis.functions)}, "
                f"fallback={len(fallback_funcs)}, size={len(out_bytes)})\n"
            )
        return 0

    except PatcherError as exc:
        _log_error(exc.message)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - safety net for malformed ELFs
        _log_error(f"unexpected failure processing '{args.input_elf}': {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
