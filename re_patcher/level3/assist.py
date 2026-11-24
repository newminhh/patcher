"""Level 3C decompilation assist scaffold generator.

Generates a human/agent-portable C scaffold for decompiler-assisted patching,
including C23 prototypes, ABI hazard audits, and build files.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .backends import (
    DecompiledFunction,
    IDAFreeBackend,
    ingest_exported_dir,
    select_backend,
)

try:
    from .. import PatcherError
except Exception:  # pragma: no cover - defensive import for standalone use
    class PatcherError(Exception):
        """Fallback patcher error."""


@dataclass
class AssistReport:
    """Summary of generated decompilation assist scaffold."""

    out_dir: str
    backend_name: str
    functions_exported: int
    hazards_found: int
    ida_free_manual: bool
    files_written: List[str] = field(default_factory=list)


@dataclass
class _NormalizedImport:
    symbol: str
    status: str
    dsl: str
    lib: str
    reason: str
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_data(self) -> bool:
        return bool(re.match(r"^D\d+$", self.dsl.strip()))

    @property
    def is_forbidden(self) -> bool:
        return self.status.upper() == "FORBIDDEN_ABI"

    @property
    def is_untyped(self) -> bool:
        return self.status.upper() == "UNTYPED"


_DSL_TYPE_MAP = {
    "v": "void",
    "i": "int32_t",
    "u": "uint32_t",
    "p": "void *",
    "q": "int64_t",
    "Q": "uint64_t",
    "f": "float",
    "d": "double",
}

_ABI_DIVERGENT_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\boff_t\b"),
        "Replace off_t with an explicit int64_t (or the x32 ABI-correct type) and audit file-offset call sites.",
    ),
    (
        re.compile(r"\btime_t\b"),
        "Replace time_t with int64_t on the wire and convert to/from native time_t at boundaries.",
    ),
    (
        re.compile(r"\bstruct\s+stat\b|\bstat\b"),
        "Use <sys/stat.h> native struct stat; x32 and i386 layouts/field sizes differ across ABIs.",
    ),
    (
        re.compile(r"\bpthread_"),
        "Audit pthread_* types/calls for ABI layout differences; prefer native pthread headers and avoid hardcoded struct sizes.",
    ),
    (
        re.compile(r"\bsigaction\b"),
        "Use native <signal.h> struct sigaction; kernel/sigaction ABI layouts differ on x32.",
    ),
    (
        re.compile(r"\bsret\b", re.IGNORECASE),
        "Hidden sret pointer ABI differs by target; return small aggregates by value or use an explicit out-pointer with verified ABI.",
    ),
)


def generate_assist_scaffold(
    elf_path: str,
    out_dir: str,
    *,
    functions: List[Tuple[int, str, Optional[List[str]]]],
    imports: List[Dict[str, Any]],
    lookup_table: Any = None,
    decompiler_choice: str = "auto",
    import_decompiled_dir: Optional[str] = None,
    target_interp: Optional[str] = None,
) -> AssistReport:
    """Generate a Level 3C human/agent decompilation assist scaffold."""
    if not isinstance(out_dir, str) or not out_dir.strip():
        raise ValueError("out_dir must be a non-empty string")
    if not isinstance(elf_path, str) or not elf_path.strip():
        raise ValueError("elf_path must be a non-empty string")

    functions = _normalize_functions(functions)
    imports = [dict(item) for item in (imports or []) if isinstance(item, dict)]

    backend, backend_cap, ida_free_manual = _select_assist_backend(
        decompiler_choice=decompiler_choice,
        import_decompiled_dir=import_decompiled_dir,
    )

    out_dir_abs = os.path.abspath(out_dir)
    _prepare_clean_layout(out_dir_abs, include_scripts=ida_free_manual)

    files_written: List[str] = []
    normalized_imports = _normalize_imports(imports, lookup_table)
    hazards, hazards_found = _collect_abi_hazards(normalized_imports, imports)

    decompiled_by_va = _resolve_decompiled_functions(
        elf_path=elf_path,
        functions=functions,
        backend=backend,
        backend_cap=backend_cap,
        import_decompiled_dir=import_decompiled_dir,
    )

    src_files = _write_function_sources(
        out_dir_abs=out_dir_abs,
        functions=functions,
        decompiled_by_va=decompiled_by_va,
        backend_name=backend_cap.name if backend_cap else (backend.name if backend else "unknown"),
        ida_free_manual=ida_free_manual,
    )
    files_written.extend(src_files)

    files_written.append(
        _write_text_file(
            os.path.join(out_dir_abs, "include", "nm_imports.h"),
            _render_nm_imports_h(normalized_imports),
        )
    )
    files_written.append(
        _write_text_file(
            os.path.join(out_dir_abs, "abi_hazards.txt"),
            _render_abi_hazards_txt(hazards),
        )
    )
    files_written.append(
        _write_text_file(
            os.path.join(out_dir_abs, "PORTING_PROMPT.md"),
            _render_porting_prompt_md(
                elf_path=elf_path,
                backend_name=backend_cap.name if backend_cap else (backend.name if backend else "unknown"),
                target_interp=target_interp,
            ),
        )
    )
    files_written.append(
        _write_text_file(
            os.path.join(out_dir_abs, "Makefile"),
            _render_makefile(normalized_imports),
        )
    )
    files_written.append(
        _write_text_file(
            os.path.join(out_dir_abs, "README.md"),
            _render_readme(
                elf_path=elf_path,
                out_dir=out_dir_abs,
                backend=backend,
                backend_cap=backend_cap,
                ida_free_manual=ida_free_manual,
                functions=functions,
                target_interp=target_interp,
            ),
        )
    )

    if ida_free_manual:
        files_written.append(_copy_ida_export_script(out_dir_abs))

    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered_files: List[str] = []
    for path in files_written:
        if path not in seen:
            seen.add(path)
            ordered_files.append(path)

    return AssistReport(
        out_dir=out_dir_abs,
        backend_name=backend_cap.name if backend_cap else (backend.name if backend else "unknown"),
        functions_exported=len(src_files),
        hazards_found=hazards_found,
        ida_free_manual=ida_free_manual,
        files_written=ordered_files,
    )


def _normalize_functions(
    functions: Sequence[Tuple[int, str, Optional[List[str]]]]
) -> List[Tuple[int, str, List[str]]]:
    normalized: List[Tuple[int, str, List[str]]] = []
    for item in functions or []:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError("Each function must be a tuple (va, name, gate_reasons?)")
        va_raw, name_raw = item[0], item[1]
        reasons_raw = item[2] if len(item) > 2 else None
        va = int(va_raw)
        name = str(name_raw).strip() or f"sub_{va:x}"
        reasons: List[str] = []
        if isinstance(reasons_raw, list):
            reasons = [str(r) for r in reasons_raw if str(r).strip()]
        elif reasons_raw is not None:
            reasons = [str(reasons_raw)]
        normalized.append((va, name, reasons))
    return normalized


def _select_assist_backend(
    *, decompiler_choice: str, import_decompiled_dir: Optional[str]
) -> Tuple[Optional[Any], Optional[Any], bool]:
    try:
        backend = select_backend(decompiler_choice, require_headless=False)
    except Exception as exc:
        if import_decompiled_dir:
            # Assist mode can still ingest exported C without a live backend.
            fallback = IDAFreeBackend()
            cap = fallback.probe()
            return fallback, cap, True
        raise

    cap = backend.probe()
    ida_free_manual = bool(getattr(cap, "assist_only", False))
    return backend, cap, ida_free_manual


def _prepare_clean_layout(out_dir_abs: str, *, include_scripts: bool) -> None:
    os.makedirs(out_dir_abs, exist_ok=True)
    for subdir in ("src", "include", "scripts"):
        path = os.path.join(out_dir_abs, subdir)
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(os.path.join(out_dir_abs, "src"), exist_ok=True)
    os.makedirs(os.path.join(out_dir_abs, "include"), exist_ok=True)
    if include_scripts:
        os.makedirs(os.path.join(out_dir_abs, "scripts"), exist_ok=True)


def _safe_func_name(name: str, va: int, used: set[str]) -> str:
    base = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip() or f"sub_{va:x}")
    base = re.sub(r"_+", "_", base).strip("_") or f"sub_{va:x}"
    if base[0].isdigit():
        base = f"fn_{base}"
    candidate = base
    suffix = 2
    while candidate.lower() in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate.lower())
    return candidate


def _resolve_decompiled_functions(
    *,
    elf_path: str,
    functions: List[Tuple[int, str, List[str]]],
    backend: Optional[Any],
    backend_cap: Optional[Any],
    import_decompiled_dir: Optional[str],
) -> Dict[int, DecompiledFunction]:
    if import_decompiled_dir:
        return ingest_exported_dir(import_decompiled_dir)

    assist_only = bool(getattr(backend_cap, "assist_only", False)) if backend_cap else False
    if assist_only or backend is None:
        return {}

    func_pairs = [(va, name) for va, name, _ in functions]
    if not func_pairs:
        return {}
    try:
        result = backend.decompile_batch(elf_path, func_pairs, timeout=120)
        if isinstance(result, dict):
            return {int(k): v for k, v in result.items() if isinstance(v, DecompiledFunction)}
    except Exception:
        # Fail loud in report via stub generation; backend issues should be visible in README/warnings.
        return {}
    return {}


def _write_function_sources(
    *,
    out_dir_abs: str,
    functions: List[Tuple[int, str, List[str]]],
    decompiled_by_va: Dict[int, DecompiledFunction],
    backend_name: str,
    ida_free_manual: bool,
) -> List[str]:
    written: List[str] = []
    used_names: set[str] = set()
    src_dir = os.path.join(out_dir_abs, "src")

    for va, name, gate_reasons in functions:
        safe_name = _safe_func_name(name, va, used_names)
        dec = decompiled_by_va.get(int(va))
        content = _render_function_c(
            va=va,
            func_name=name,
            safe_name=safe_name,
            gate_reasons=gate_reasons,
            decompiled=dec,
            backend_name=backend_name,
            ida_free_manual=ida_free_manual,
        )
        path = os.path.join(src_dir, f"{safe_name}.c")
        written.append(_write_text_file(path, content))

    return written


def _render_function_c(
    *,
    va: int,
    func_name: str,
    safe_name: str,
    gate_reasons: List[str],
    decompiled: Optional[DecompiledFunction],
    backend_name: str,
    ida_free_manual: bool,
) -> str:
    warnings = list(getattr(decompiled, "warnings", []) or []) if decompiled else []
    backend = getattr(decompiled, "backend", backend_name) if decompiled else backend_name
    gate_reason_str = "; ".join(gate_reasons) if gate_reasons else "(none)"
    warning_str = "; ".join(str(w) for w in warnings) if warnings else "(none)"

    header = [
        "/*",
        f" * Function: {func_name}",
        f" * Safe name: {safe_name}",
        f" * VA: 0x{va:x}",
        f" * Backend: {backend}",
        f" * Warnings: {warning_str}",
        f" * Gate reasons: {gate_reason_str}",
        " */",
        '#include "nm_imports.h"',
        "",
    ]

    if decompiled and str(getattr(decompiled, "c_code", "")).strip():
        body = str(decompiled.c_code).rstrip() + "\n"
    else:
        body = _render_stub_body(
            va=va,
            func_name=func_name,
            safe_name=safe_name,
            ida_free_manual=ida_free_manual,
        )

    return "\n".join(header) + "\n" + body


def _render_stub_body(*, va: int, func_name: str, safe_name: str, ida_free_manual: bool) -> str:
    if ida_free_manual:
        todo = [
            f"// TODO: Awaiting manual IDA Free export for {func_name} @ 0x{va:x}.",
            "// TODO: Export with scripts/ida_export_c.py or File -> Produce file -> Create C file,",
            "// then re-run newminhh-patch --decompile-assist <out_dir> --import-decompiled <exported_dir> ...",
            f"// TODO: Replace this stub with the real decompiled/repaired implementation of {func_name}.",
        ]
    else:
        todo = [
            f"// TODO: No decompiled body available for {func_name} @ 0x{va:x}.",
            "// TODO: Inspect disassembly and port the function manually.",
        ]

    return (
        "\n".join(todo)
        + "\n\n"
        + f"/* Assembly/disassembly TODO: recover function at VA 0x{va:x} ({func_name}). */\n"
        + f"void {safe_name}(void) {{\n"
        + "    /* Stub placeholder; replace with correct signature and implementation. */\n"
        + "}\n"
    )


def _normalize_imports(imports: List[Dict[str, Any]], lookup_table: Any) -> List[_NormalizedImport]:
    records: Dict[str, _NormalizedImport] = {}

    def upsert(rec: _NormalizedImport) -> None:
        key = rec.symbol or f"<unknown:{len(records)}>"
        existing = records.get(key)
        if existing is None:
            records[key] = rec
            return
        # Merge: prefer explicit status/dsl/lib/reason when present.
        if rec.status and rec.status.upper() != "UNKNOWN":
            existing.status = rec.status
        if rec.dsl and not existing.dsl:
            existing.dsl = rec.dsl
        if rec.lib and not existing.lib:
            existing.lib = rec.lib
        if rec.reason and not existing.reason:
            existing.reason = rec.reason
        merged_raw = dict(existing.raw)
        merged_raw.update(rec.raw)
        existing.raw = merged_raw

    for raw in _iter_lookup_records(lookup_table):
        rec = _import_from_mapping(raw, default_status="UNKNOWN")
        if rec.symbol:
            upsert(rec)

    for raw in imports:
        rec = _import_from_mapping(raw, default_status="UNKNOWN")
        if rec.symbol:
            upsert(rec)

    # Stable sort: status then symbol
    status_order = {"FORBIDDEN_ABI": 0, "UNTYPED": 1, "COMPAT": 2, "NATIVE_TYPED": 3, "UNKNOWN": 4}
    return sorted(records.values(), key=lambda r: (status_order.get(r.status.upper(), 99), r.symbol))


def _iter_lookup_records(lookup_table: Any) -> Iterable[Dict[str, Any]]:
    if lookup_table is None:
        return []

    if isinstance(lookup_table, dict):
        # Possible shapes: {symbol: spec}, {"imports"/"entries"/"bindings": ...}, or a single record
        for key in ("imports", "entries", "bindings", "symbols", "records"):
            val = lookup_table.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
            if isinstance(val, dict):
                return _dict_entries_to_records(val)
        # If dict values look like records/specs, treat as symbol->spec mapping
        if lookup_table and all(isinstance(k, str) for k in lookup_table.keys()):
            maybe_records = [v for v in lookup_table.values() if isinstance(v, dict)]
            if maybe_records or any(isinstance(v, (str, tuple, list)) for v in lookup_table.values()):
                return _dict_entries_to_records(lookup_table)
        # Single record dict
        if "symbol" in lookup_table or "name" in lookup_table:
            return [lookup_table]
        return []

    if isinstance(lookup_table, list):
        return [x for x in lookup_table if isinstance(x, dict)]

    # Object with common attributes/methods
    for attr in ("imports", "entries", "bindings", "records", "symbols"):
        if hasattr(lookup_table, attr):
            val = getattr(lookup_table, attr)
            if callable(val):
                try:
                    val = val()
                except Exception:
                    val = None
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
            if isinstance(val, dict):
                return _dict_entries_to_records(val)

    if hasattr(lookup_table, "items"):
        try:
            items = dict(lookup_table.items())  # type: ignore[arg-type]
            if isinstance(items, dict):
                return _dict_entries_to_records(items)
        except Exception:
            pass

    try:
        seq = list(lookup_table)
    except Exception:
        return []
    return [x for x in seq if isinstance(x, dict)]


def _dict_entries_to_records(mapping: Dict[Any, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for key, value in mapping.items():
        symbol = str(key)
        if isinstance(value, dict):
            rec = dict(value)
            rec.setdefault("symbol", symbol)
            records.append(rec)
        elif isinstance(value, str):
            records.append({"symbol": symbol, "dsl": value, "signature": value})
        elif isinstance(value, (tuple, list)):
            rec = {"symbol": symbol}
            if len(value) >= 1 and isinstance(value[0], str):
                rec["status"] = value[0]
            if len(value) >= 2 and isinstance(value[1], str):
                rec["dsl"] = value[1]
            if len(value) >= 3 and isinstance(value[2], str):
                rec["lib"] = value[2]
            if len(value) >= 4 and isinstance(value[3], str):
                rec["reason"] = value[3]
            records.append(rec)
        else:
            records.append({"symbol": symbol, "value": value})
    return records


def _import_from_mapping(raw: Dict[str, Any], *, default_status: str) -> _NormalizedImport:
    symbol = str(
        raw.get("symbol")
        or raw.get("name")
        or raw.get("import")
        or raw.get("callee")
        or ""
    ).strip()
    status = str(
        raw.get("status")
        or raw.get("kind")
        or raw.get("class")
        or raw.get("binding")
        or default_status
    ).strip().upper() or default_status
    dsl = str(
        raw.get("dsl")
        or raw.get("signature")
        or raw.get("sig")
        or raw.get("proto")
        or raw.get("type")
        or ""
    ).strip()
    lib = str(
        raw.get("lib")
        or raw.get("library")
        or raw.get("soname")
        or raw.get("provider")
        or ""
    ).strip()
    reason = str(
        raw.get("reason")
        or raw.get("note")
        or raw.get("why")
        or raw.get("gate_reason")
        or ""
    ).strip()

    # Normalize common aliases
    alias = {
        "FORBIDDEN": "FORBIDDEN_ABI",
        "FORBIDDEN_ABI": "FORBIDDEN_ABI",
        "NATIVE": "NATIVE_TYPED",
        "NATIVE_TYPED": "NATIVE_TYPED",
        "COMPAT": "COMPAT",
        "UNTYPED": "UNTYPED",
    }
    status = alias.get(status, status)

    return _NormalizedImport(
        symbol=symbol,
        status=status,
        dsl=dsl,
        lib=lib,
        reason=reason,
        raw=dict(raw),
    )


def _dsl_to_c_type(token: str) -> str:
    return _DSL_TYPE_MAP.get(token, "int32_t")


def _render_dsl_prototype(rec: _NormalizedImport) -> str:
    sym = rec.symbol or "unknown_symbol"
    dsl = rec.dsl.strip()
    status = rec.status.upper() or "UNKNOWN"
    comment = f"/* {status}"
    if rec.lib:
        comment += f" lib={rec.lib}"
    if rec.reason:
        comment += f" reason={rec.reason}"
    comment += " */"

    if not dsl:
        return f"{comment}\n/* TODO: missing DSL/signature for {sym} */\nextern int {sym}(void);"

    # Data symbol: D<n>
    m_data = re.fullmatch(r"D(\d+)", dsl)
    if m_data:
        size = m_data.group(1)
        return f"{comment}\n/* data symbol size={size} */\nextern uintptr_t {sym};"

    # Function signature DSL: first char return, rest args; ... for varargs
    ret_token = dsl[0]
    ret_type = _dsl_to_c_type(ret_token)
    args_spec = dsl[1:]
    params: List[str] = []
    i = 0
    arg_index = 0
    varargs = False
    while i < len(args_spec):
        ch = args_spec[i]
        if args_spec.startswith("...", i):
            varargs = True
            i += 3
            continue
        if ch == "D" and i + 1 < len(args_spec) and args_spec[i + 1 :].lstrip("0123456789") != args_spec[i + 1 :]:
            # Unexpected data token in arg position; treat as pointer-sized placeholder.
            j = i + 1
            while j < len(args_spec) and args_spec[j].isdigit():
                j += 1
            params.append(f"uintptr_t arg{arg_index}")
            arg_index += 1
            i = j
            continue
        c_type = _dsl_to_c_type(ch)
        params.append(f"{c_type} arg{arg_index}")
        arg_index += 1
        i += 1

    if not params:
        param_str = "void"
    else:
        param_str = ", ".join(params)
        if varargs:
            param_str += ", ..."

    return f"{comment}\nextern {ret_type} {sym}({param_str});"


def _render_nm_imports_h(imports: List[_NormalizedImport]) -> str:
    lines: List[str] = [
        "#ifndef NM_IMPORTS_H",
        "#define NM_IMPORTS_H",
        "",
        "#include <stdbool.h>",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#include <sys/types.h>",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "/* Auto-generated by newminhh --decompile-assist (Level 3C).",
        " * DSL mapping: v->void, i->int32_t, u->uint32_t, p->void*, q->int64_t,",
        " * Q->uint64_t, f->float, d->double, ...->varargs, D<n>->extern uintptr_t data symbol.",
        " */",
        "",
    ]

    if not imports:
        lines.append("/* No bound imports were provided. */")
    else:
        for rec in imports:
            lines.append(_render_dsl_prototype(rec))
            lines.append("")

    lines.extend(
        [
            "#ifdef __cplusplus",
            "}",
            "#endif",
            "",
            "#endif /* NM_IMPORTS_H */",
            "",
        ]
    )
    return "\n".join(lines)


def _collect_abi_hazards(
    normalized_imports: List[_NormalizedImport], raw_imports: List[Dict[str, Any]]
) -> Tuple[List[str], int]:
    hazards: List[str] = []

    def add(line: str) -> None:
        hazards.append(line.rstrip())

    # Audit normalized/bound imports
    for rec in normalized_imports:
        status = rec.status.upper()
        symbol = rec.symbol or "<unknown>"
        dsl = rec.dsl or "<no-dsl>"
        lib = rec.lib or "<unknown-lib>"
        base = f"IMPORT {symbol} [{status}] lib={lib} dsl={dsl}"
        lower_blob = " ".join([symbol, status, dsl, rec.reason, json.dumps(rec.raw, sort_keys=True, default=str)]).lower()

        if status == "FORBIDDEN_ABI":
            add(
                base
                + " :: FORBIDDEN_ABI import cannot be called directly. "
                + "Porting advice: replace with a native x32-compatible equivalent or isolate behind a manually-audited shim; "
                + (rec.reason or "inspect ABI/layout/calling convention before use.")
            )

        if status == "UNTYPED":
            add(
                base
                + " :: UNTYPED import has no trusted signature. "
                + "Porting advice: recover prototype from authoritative headers/docs, verify argument widths/return ABI, then update lookup metadata."
            )

        if "sret" in lower_blob:
            add(
                base
                + " :: hidden sret ABI hazard detected. "
                + "Porting advice: hidden struct-return pointer ABI differs by target; return small aggregates by value or use explicit out-pointer with verified ABI."
            )

        for pat, advice in _ABI_DIVERGENT_PATTERNS:
            if pat.search(lower_blob):
                # avoid duplicate sret line from generic patterns? sret pattern included above; keep generic but no double
                if pat.pattern == r"\bsret\b" and "hidden sret ABI hazard detected" in hazards[-1] if hazards else False:
                    continue
                add(base + f" :: ABI-divergent symbol/pattern '{pat.pattern}' detected. Porting advice: {advice}")

    # Audit raw imports/call sites for call-site hints
    for raw in raw_imports:
        symbol = str(raw.get("symbol") or raw.get("name") or raw.get("callee") or "<unknown>")
        status = str(raw.get("status") or raw.get("kind") or "UNKNOWN").upper()
        callsites = raw.get("call_sites") or raw.get("callsites") or raw.get("sites") or []
        blob = json.dumps(raw, sort_keys=True, default=str)
        lower_blob = blob.lower()

        has_divergent = status in {"FORBIDDEN_ABI", "UNTYPED"} or "sret" in lower_blob or any(
            pat.search(lower_blob) for pat, _ in _ABI_DIVERGENT_PATTERNS
        )
        if not has_divergent:
            continue

        if isinstance(callsites, list) and callsites:
            for site in callsites:
                site_str = site if isinstance(site, str) else json.dumps(site, sort_keys=True, default=str)
                add(
                    f"CALLSITE {symbol} [{status}] @ {site_str} :: audit required. "
                    "Porting advice: verify argument/register mapping, struct layout, and ABI-sensitive types at this call site."
                )
        else:
            # Ensure we have at least one line when raw record had hazard markers but wasn't normalized
            if not any(line.startswith(f"IMPORT {symbol} [") for line in hazards):
                add(
                    f"IMPORT {symbol} [{status}] :: hazard metadata detected in raw import. "
                    "Porting advice: verify signature, ABI, and struct/time/pthread/sret-sensitive usage."
                )

    # Deduplicate preserving order
    deduped: List[str] = []
    seen: set[str] = set()
    for line in hazards:
        if line not in seen:
            seen.add(line)
            deduped.append(line)
    return deduped, len(deduped)


def _render_abi_hazards_txt(hazards: List[str]) -> str:
    header = [
        "# ABI Hazards Audit (Level 3C)",
        "#",
        "# This file audits imports/call sites touching FORBIDDEN_ABI, UNTYPED, sret,",
        "# off_t/time_t/struct stat/pthread_*/sigaction ABI-divergent symbols.",
        "#",
    ]
    if not hazards:
        header.append("No ABI hazards detected from provided imports/lookup metadata.")
        header.append("")
        return "\n".join(header)
    header.append(f"Total hazards: {len(hazards)}")
    header.append("")
    body = [f"{idx + 1}. {line}" for idx, line in enumerate(hazards)]
    return "\n".join(header + body) + "\n"


def _detect_native_lib_flags(imports: List[_NormalizedImport]) -> List[str]:
    flags: List[str] = []
    seen: set[str] = set()
    for rec in imports:
        lib = (rec.lib or "").strip()
        if not lib:
            continue
        # Extract -l flag from common soname patterns: libfoo.so, libfoo.so.1, foo
        base = os.path.basename(lib)
        m = re.match(r"^lib([A-Za-z0-9_.+\-]+)\.so(?:\.[0-9A-Za-z_.\-]+)?$", base)
        if m:
            name = m.group(1)
            flag = f"-l{name}"
        elif base.endswith(".a") and base.startswith("lib"):
            flag = f"-l{base[3:-2]}"
        elif re.fullmatch(r"[A-Za-z0-9_.+\-]+", base):
            flag = f"-l{base}" if not base.startswith("-l") else base
        else:
            continue
        if flag not in seen:
            seen.add(flag)
            flags.append(flag)
    return flags


def _render_makefile(imports: List[_NormalizedImport]) -> str:
    lib_flags = _detect_native_lib_flags(imports)
    libs_line = " ".join(lib_flags)
    return (
        "CC ?= clang\n"
        "TARGET_FLAGS ?= -target x86_64-linux-gnux32\n"
        "X32_SYSROOT ?= /home/shadichy/newminhh-x32-sysroot\n"
        "CFLAGS = -std=gnu23 -O2 -Wall -Wextra -Iinclude $(TARGET_FLAGS) --sysroot=$(X32_SYSROOT)\n"
        "LDFLAGS = $(TARGET_FLAGS) --sysroot=$(X32_SYSROOT)\n"
        f"LIBS = {libs_line}\n"
        "\n"
        "SRCS := $(wildcard src/*.c)\n"
        "OBJS := $(SRCS:.c=.o)\n"
        "OUT ?= assist_objects.a\n"
        "PROGRAM ?= assist_bundle\n"
        "\n"
        ".PHONY: all clean link\n"
        "\n"
        "all: $(OUT)\n"
        "\n"
        "$(OUT): $(OBJS)\n"
        "\t$(AR) rcs $@ $(OBJS)\n"
        "\n"
        "link: $(OBJS)\n"
        "\t$(CC) $(LDFLAGS) -o $(PROGRAM) $(OBJS) $(LIBS)\n"
        "\n"
        "src/%.o: src/%.c\n"
        "\t$(CC) $(CFLAGS) -c $< -o $@\n"
        "\n"
        "clean:\n"
        "\trm -f $(OBJS) $(OUT) $(PROGRAM)\n"
    )


def _render_porting_prompt_md(*, elf_path: str, backend_name: str, target_interp: Optional[str]) -> str:
    interp_line = target_interp or "(not specified)"
    return f"""# Level 3C Porting Prompt

Use this prompt with an LLM coding agent to repair decompiler artifacts in this scaffold.

## Context

- Binary: `{elf_path}`
- Backend: `{backend_name}`
- Target interpreter: `{interp_line}`
- Scaffold: `src/*.c`, `include/nm_imports.h`, `abi_hazards.txt`, `Makefile`

## Task

Repair the decompiled C functions so they compile cleanly for x32 and preserve original behavior.

## Required fixes

1. **Decompiler artifacts**: remove invalid register artifacts, undefined pseudo-types, and non-C constructs; keep semantics.
2. **PIC/string constants**: fix constant-folded PIC/string references; do not embed absolute addresses as integer literals.
3. **IPA-RA/register args**: repair function signatures and callsites where decompiler inferred wrong register/argument passing.
4. **ABI/layout hazards**: fix `struct stat`, `off_t`, `time_t`, `pthread_*`, `sigaction`, hidden `sret`, and other ABI-divergent uses. Use native headers/layouts for the x32 sysroot.
5. **Imports**: use prototypes from `include/nm_imports.h`; resolve `UNTYPED`/`FORBIDDEN_ABI` imports with safe x32-compatible alternatives.
6. **Build cleanliness**: ensure every `src/*.c` compiles with:

```sh
make X32_SYSROOT=/path/to/x32-sysroot
```

The Makefile defaults to `-std=gnu23 -O2 -Wall -Wextra -Iinclude -target x86_64-linux-gnux32 --sysroot=$(X32_SYSROOT)` (or equivalent `-mx32` x32 toolchain flags).

## Output expectations

- Keep function behavior equivalent to the original binary.
- Add concise comments only where ABI/layout/porting decisions are non-obvious.
- Do not introduce host-only assumptions; target is Linux x32.
- Report any unresolved ABI hazards from `abi_hazards.txt`.
"""


def _render_readme(
    *,
    elf_path: str,
    out_dir: str,
    backend: Optional[Any],
    backend_cap: Optional[Any],
    ida_free_manual: bool,
    functions: List[Tuple[int, str, List[str]]],
    target_interp: Optional[str],
) -> str:
    backend_name = getattr(backend_cap, "name", None) or (backend.name if backend else "unknown")
    available = getattr(backend_cap, "available", None)
    headless = getattr(backend_cap, "headless", None)
    assist_only = getattr(backend_cap, "assist_only", None)
    cap_path = getattr(backend_cap, "path", None)
    reason = getattr(backend_cap, "reason", "")
    interp = target_interp or "(not specified)"

    lines: List[str] = [
        "# Level 3C Decompile Assist Scaffold",
        "",
        "This directory was generated by `newminhh-patch --decompile-assist` (Level 3C human/agent porting scaffold).",
        "",
        "## Binary",
        "",
        f"- ELF: `{elf_path}`",
        f"- Target interpreter: `{interp}`",
        f"- Output dir: `{out_dir}`",
        "",
        "## Selected backend",
        "",
        f"- Name: `{backend_name}`",
        f"- Available: `{available}`",
        f"- Headless: `{headless}`",
        f"- Assist-only: `{assist_only}`",
        f"- Path: `{cap_path}`",
        f"- Reason: {reason}",
        "",
        "## Build instructions (x32 sysroot)",
        "",
        "Build against an x32 sysroot with the provided Makefile:",
        "",
        "```sh",
        "make X32_SYSROOT=/path/to/x32-sysroot",
        "```",
        "",
        "Defaults:",
        "- `CC ?= clang`",
        "- `TARGET_FLAGS ?= -target x86_64-linux-gnux32`",
        "- `X32_SYSROOT ?= /home/shadichy/newminhh-x32-sysroot`",
        "- `CFLAGS = -std=gnu23 -O2 -Wall -Wextra -Iinclude $(TARGET_FLAGS) --sysroot=$(X32_SYSROOT)`",
        "",
        "## Functions",
        "",
        "| Function | VA | Level 3A gate reasons |",
        "|---|---:|---|",
    ]

    for va, name, gate_reasons in functions:
        reasons = "<br>".join(_md_escape(r) for r in gate_reasons) if gate_reasons else "(none)"
        lines.append(f"| `{_md_escape(name)}` | `0x{va:x}` | {reasons} |")

    if not functions:
        lines.append("| _(none)_ | - | - |")

    if ida_free_manual:
        lines.extend(
            [
                "",
                "## IDA Free manual export",
                "",
                "IDA Free was selected (assist-only/non-headless). Use one of the methods below to export decompiled C, then re-import it.",
                "",
                "### Method 1: IDAPython script (IDA Free 8.x/9.x)",
                "",
                "1. Open the target binary in IDA Free 8.x/9.x and wait for analysis to complete.",
                "2. Run `File -> Script file...` and select `scripts/ida_export_c.py` from this scaffold directory.",
                "3. Choose an export directory when prompted (or set `IDA_EXPORT_DIR`). The script writes one `.c` file per function and `manifest.json`.",
                "4. Re-run newminhh-patch to ingest the exported C:",
                "",
                "```sh",
                "newminhh-patch --decompile-assist <out_dir> --import-decompiled <exported_dir> ...",
                "```",
                "",
                "### Method 2: Manual fallback (older IDA Free builds without IDAPython)",
                "",
                "1. In IDA Free, use `File -> Produce file -> Create C file...` to generate a C file.",
                "2. If needed, split or organize exported functions into an export directory compatible with `--import-decompiled` (include `manifest.json` when possible).",
                "3. Re-run newminhh-patch to ingest the exported C:",
                "",
                "```sh",
                "newminhh-patch --decompile-assist <out_dir> --import-decompiled <exported_dir> ...",
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `src/*.c`: per-function decompiled/stub C sources.",
            "- `include/nm_imports.h`: C23 prototypes/declarations for bound imports.",
            "- `abi_hazards.txt`: ABI hazard audit with porting advice.",
            "- `PORTING_PROMPT.md`: LLM agent prompt for repairing decompiler artifacts.",
            "- `Makefile`: x32 sysroot build.",
        ]
    )
    if ida_free_manual:
        lines.append("- `scripts/ida_export_c.py`: IDA Free export helper (copied into scaffold).")
    lines.append("")
    return "\n".join(lines)


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def _copy_ida_export_script(out_dir_abs: str) -> str:
    src_path = os.path.join(os.path.dirname(__file__), "scripts", "ida_export_c.py")
    dst_dir = os.path.join(out_dir_abs, "scripts")
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, "ida_export_c.py")
    if not os.path.isfile(src_path):
        # Fallback: write a minimal notice; but prefer failing loud? Better fail loud.
        raise FileNotFoundError(f"IDA export script not found: {src_path}")
    shutil.copyfile(src_path, dst_path)
    return dst_path


def _write_text_file(path: str, content: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


__all__ = ["AssistReport", "generate_assist_scaffold"]
