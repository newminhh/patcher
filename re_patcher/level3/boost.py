"""Level 3A gated per-function decompilation, freestanding -mx32 recompilation,
differential verification, and guest-cdecl <-> x32-SysV boundary stub generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Set, Tuple

from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection
from elftools.elf.sections import SymbolTableSection

from .backends import DecompiledFunction, select_backend
from .diffverify import verify_boosted_function
from .gates import GateVerdict, evaluate_purity_gate
from .sanitize import FunctionSignature, sanitize_function

__all__ = [
    "BoostedFunction",
    "FunctionBoostOutcome",
    "Level3BoostReport",
    "compile_freestanding_x32",
    "generate_cdecl_to_x32_boundary_stub",
    "run_level3_boost",
]

# Standard pure libm symbols that may appear as undefined relocations if the
# purity gate explicitly permitted a PURE NATIVE_TYPED math import.
DEFAULT_PURE_LIBM_EXTERNS: frozenset[str] = frozenset({
    "acos", "acosf", "asin", "asinf", "atan", "atanf", "atan2", "atan2f",
    "cbrt", "cbrtf", "ceil", "ceilf", "cos", "cosf", "cosh", "coshf",
    "exp", "expf", "exp2", "exp2f", "fabs", "fabsf", "floor", "floorf",
    "fma", "fmaf", "fmax", "fmaxf", "fmin", "fminf", "fmod", "fmodf",
    "hypot", "hypotf", "log", "logf", "log10", "log10f", "log2", "log2f",
    "pow", "powf", "round", "roundf", "sin", "sinf", "sinh", "sinhf",
    "sqrt", "sqrtf", "tan", "tanf", "tanh", "tanhf", "trunc", "truncf",
})

_FORBIDDEN_EXTERN_SYMBOLS: frozenset[str] = frozenset({
    "__stack_chk_fail",
    "__stack_chk_fail_local",
    "__chk_fail",
})

_ALLOWED_ELF_SECTIONS: frozenset[str] = frozenset({
    "",
    ".text",
    ".comment",
    ".note.GNU-stack",
    ".note.gnu.property",
    ".symtab",
    ".strtab",
    ".shstrtab",
    ".rela.text",
    ".rel.text",
})

_GP_ARG_REGS_32: Tuple[str, ...] = ("edi", "esi", "edx", "ecx", "r8d", "r9d")
_GP_ARG_REGS_64: Tuple[str, ...] = ("rdi", "rsi", "rdx", "rcx", "r8", "r9")


@dataclass
class BoostedFunction:
    """Verified freestanding -mx32 replacement for a pure guest i386 function."""

    name: str
    va: int
    size: int
    ret_type: str
    arg_types: List[str]
    backend: str
    c_source: str
    obj_bytes: bytes
    text_bytes: bytes
    boundary_stub_asm: str
    diff_mode: str
    trials_run: int


@dataclass
class FunctionBoostOutcome:
    """Per-function Level 3A boost attempt outcome (boosted or fallback reason)."""

    name: str
    va: int
    boosted: bool
    stage_failed: Optional[str]
    reason: str
    gate_reasons: List[str]
    boosted_func: Optional[BoostedFunction] = None


@dataclass
class Level3BoostReport:
    """Aggregate report across all candidate functions evaluated by Level 3A."""

    backend: str
    total_functions: int
    gate_passed: int
    boosted_count: int
    outcomes: List[FunctionBoostOutcome] = field(default_factory=list)
    boosted_map: Dict[int, BoostedFunction] = field(default_factory=dict)


def _safe_c_name(func_name: str, va: int = 0) -> str:
    """Convert a symbol name into a valid C identifier suffix."""
    raw = (func_name or "").strip()
    if raw.startswith("nm_l3_"):
        raw = raw[len("nm_l3_"):]
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = f"sub_{va:x}"
    if cleaned[0].isdigit():
        cleaned = f"fn_{cleaned}"
    return cleaned


def _is_pic_thunk(func_name: str, func_bytes: bytes) -> bool:
    """Return True for x86 PIC get_pc_thunk helpers whose side effect is in a register."""
    if func_name.startswith("__x86.get_pc_thunk."):
        return True
    # 4-byte `mov <reg32>, (%esp); ret` (8B 04|0C|14|1C|2C|34|3C 24 C3)
    if (
        len(func_bytes) == 4
        and func_bytes[0] == 0x8B
        and (func_bytes[1] & 0xC7) == 0x04
        and func_bytes[2] == 0x24
        and func_bytes[3] == 0xC3
    ):
        return True
    return False


def generate_cdecl_to_x32_boundary_stub(
    func_name: str,
    va: int,
    ret_type: str,
    arg_types: List[str],
) -> str:
    """Generate an AT&T x32 GAS boundary stub bridging 4-byte guest cdecl -> x32 SysV.

    Guest entry stack:
        0(%rsp): 4-byte guest return VA
        4(%rsp): arg0 (4B slot; 8B types occupy 2 consecutive slots)

    The stub preserves i386 callee-saved %rsi/%rdi (which are caller-saved in x32),
    aligns %rsp to 16 bytes, marshals scalar arguments into x32 SysV registers,
    calls ``nm_l3_<safe_name>``, converts ``i64``/``u64`` returns to ``%edx:%eax``
    and ``f32``/``f64`` returns to x87 ``ST(0)``, restores %rsi/%rdi/%rbp/%rsp,
    and returns via a 4-byte guest return pop.
    """
    safe_name = _safe_c_name(func_name, va)
    entry_label = f"__nm_l3_entry_{va:x}"
    target_sym = f"nm_l3_{safe_name}"

    lines: List[str] = [
        "    .text",
        f"    .globl {entry_label}",
        f"    .type {entry_label}, @function",
        f"{entry_label}:",
        "    pushq %rbp",
        "    movq %rsp, %rbp",
        "    pushq %rsi",
        "    pushq %rdi",
        "    andq $-16, %rsp",
    ]

    # Marshal cdecl stack arguments starting at 12(%rbp) (original 4(%rsp)).
    gp_idx = 0
    xmm_idx = 0
    stack_off = 12
    for raw_t in arg_types:
        t = str(raw_t).strip()
        if t in ("i64", "u64"):
            if gp_idx >= len(_GP_ARG_REGS_64):
                raise ValueError(f"Too many GP arguments for x32 SysV stub: {arg_types}")
            lines.append(f"    movq {stack_off}(%rbp), %{_GP_ARG_REGS_64[gp_idx]}")
            gp_idx += 1
            stack_off += 8
        elif t == "f32":
            if xmm_idx >= 8:
                raise ValueError(f"Too many SSE float arguments for x32 SysV stub: {arg_types}")
            lines.append(f"    movss {stack_off}(%rbp), %xmm{xmm_idx}")
            xmm_idx += 1
            stack_off += 4
        elif t == "f64":
            if xmm_idx >= 8:
                raise ValueError(f"Too many SSE double arguments for x32 SysV stub: {arg_types}")
            lines.append(f"    movsd {stack_off}(%rbp), %xmm{xmm_idx}")
            xmm_idx += 1
            stack_off += 8
        else:
            # i32, u32, ptr_const_bytes -> 32-bit GP register
            if gp_idx >= len(_GP_ARG_REGS_32):
                raise ValueError(f"Too many GP arguments for x32 SysV stub: {arg_types}")
            lines.append(f"    movl {stack_off}(%rbp), %{_GP_ARG_REGS_32[gp_idx]}")
            gp_idx += 1
            stack_off += 4

    lines.append(f"    call {target_sym}")

    # Return value conversion from x32 SysV -> i386 cdecl.
    rt = str(ret_type).strip()
    if rt in ("i64", "u64"):
        lines.extend([
            "    movq %rax, %rdx",
            "    shrq $32, %rdx",
        ])
    elif rt == "f32":
        lines.extend([
            "    subq $16, %rsp",
            "    movss %xmm0, (%rsp)",
            "    flds (%rsp)",
            "    addq $16, %rsp",
        ])
    elif rt == "f64":
        lines.extend([
            "    subq $16, %rsp",
            "    movsd %xmm0, (%rsp)",
            "    fldl (%rsp)",
            "    addq $16, %rsp",
        ])

    # Restore callee-saved registers and guest %rsp, then pop 4-byte return VA.
    lines.extend([
        "    movq -16(%rbp), %rdi",
        "    movq -8(%rbp), %rsi",
        "    leaq 8(%rbp), %rsp",
        "    movq 0(%rbp), %rbp",
        "    movl (%rsp), %r11d",
        "    leaq 4(%rsp), %rsp",
        "    jmp *%r11",
        f"    .size {entry_label}, .-{entry_label}",
        "",
    ])
    return "\n".join(lines)


def _select_compiler(compiler: Optional[str]) -> Optional[str]:
    if compiler:
        return shutil.which(compiler) or (compiler if os.path.isfile(compiler) else None)
    for cand in ("gcc", "clang"):
        found = shutil.which(cand)
        if found:
            return found
    return None


def _validate_x32_object(
    obj_path: str,
    expected_symbol: str,
    allowed_pure_externs: Set[str],
) -> Tuple[Optional[bytes], str]:
    """Validate compiled -mx32 ELF object and extract .text bytes."""
    try:
        with open(obj_path, "rb") as fh:
            elf = ELFFile(fh)
            if elf.elfclass != 32 or elf["e_machine"] != "EM_X86_64":
                return (
                    None,
                    f"unexpected ELF format: class={elf.elfclass}, e_machine={elf['e_machine']}",
                )

            text_bytes: Optional[bytes] = None
            symtab_sec: Optional[SymbolTableSection] = None

            for sec in elf.iter_sections():
                sname = sec.name
                sh_size = int(sec["sh_size"])
                if sname == ".text":
                    text_bytes = sec.data()
                    continue
                if isinstance(sec, SymbolTableSection):
                    symtab_sec = sec
                    continue
                if sname.startswith((".rodata", ".data", ".bss", ".tdata", ".tbss")):
                    if sh_size != 0:
                        return (
                            None,
                            f"unexpected non-empty section '{sname}' ({sh_size} bytes)",
                        )
                    continue
                if sname in _ALLOWED_ELF_SECTIONS or sname.startswith(".note."):
                    continue
                if sname.startswith((".rela.", ".rel.")) and sname != ".rela.text":
                    return None, f"unexpected relocation section '{sname}'"
                if sh_size != 0:
                    return None, f"unexpected section '{sname}' ({sh_size} bytes)"

            if not text_bytes:
                return None, "missing or empty .text section"
            if symtab_sec is None:
                return None, "missing .symtab section"

            found_expected = False
            undef_syms: Set[str] = set()
            for sym in symtab_sec.iter_symbols():
                sym_name = sym.name
                shndx = sym["st_shndx"]
                if sym_name == expected_symbol and shndx != "SHN_UNDEF":
                    found_expected = True
                if shndx == "SHN_UNDEF" and sym_name:
                    if sym_name in _FORBIDDEN_EXTERN_SYMBOLS:
                        return None, f"forbidden external symbol '{sym_name}'"
                    if sym_name not in allowed_pure_externs:
                        return None, f"unexpected undefined external symbol '{sym_name}'"
                    undef_syms.add(sym_name)

            if not found_expected:
                return None, f"compiled object does not define expected symbol '{expected_symbol}'"

            for sec in elf.iter_sections():
                if not isinstance(sec, RelocationSection):
                    continue
                if sec.name != ".rela.text":
                    return None, f"unexpected relocation section '{sec.name}'"
                rel_symtab = elf.get_section(sec["sh_link"])
                for rel in sec.iter_relocations():
                    sym_idx = int(rel["r_info_sym"])
                    rsym = rel_symtab.get_symbol(sym_idx)
                    rname = rsym.name
                    if not rname or rname not in undef_syms:
                        return (
                            None,
                            f"unexpected relocation in .text against '{rname or f'sym#{sym_idx}'}'",
                        )

            return text_bytes, ""
    except Exception as exc:
        return None, f"failed inspecting compiled object: {exc}"


def compile_freestanding_x32(
    c_source: str,
    func_name: str,
    *,
    compiler: Optional[str] = None,
    allowed_pure_externs: Optional[Set[str]] = None,
) -> Tuple[Optional[str], Optional[bytes], str]:
    """Compile sanitized C source with freestanding -mx32 flags and verify ELF .o cleanliness.

    Returns ``(obj_path_in_tmpdir, text_bytes, error_msg)``. On success, the caller
    is responsible for cleaning up ``os.path.dirname(obj_path_in_tmpdir)`` when finished.
    """
    cc = _select_compiler(compiler)
    if not cc:
        return None, None, "no suitable C compiler (gcc/clang) found in PATH"

    safe_name = _safe_c_name(func_name)
    expected_sym = func_name if func_name.startswith("nm_l3_") else f"nm_l3_{safe_name}"

    permitted_externs = set(DEFAULT_PURE_LIBM_EXTERNS)
    if allowed_pure_externs:
        permitted_externs.update(allowed_pure_externs)

    tmpdir = tempfile.mkdtemp(prefix="nm_l3_cc_")
    c_path = os.path.join(tmpdir, f"{safe_name}.c")
    obj_path = os.path.join(tmpdir, f"{safe_name}.o")

    try:
        with open(c_path, "w", encoding="utf-8") as fh:
            fh.write(c_source)

        cmd = [
            cc,
            "-mx32",
            "-ffreestanding",
            "-fno-stack-protector",
            "-fcf-protection=none",
            "-fno-asynchronous-unwind-tables",
            "-fPIC",
            "-O2",
            "-Wall",
            "-Werror",
            "-c",
            c_path,
            "-o",
            obj_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        if proc.returncode != 0 or not os.path.isfile(obj_path):
            err = (proc.stderr or proc.stdout or f"compiler exited {proc.returncode}").strip()
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None, None, f"freestanding -mx32 compile failed: {err[:600]}"

        text_bytes, val_err = _validate_x32_object(obj_path, expected_sym, permitted_externs)
        if text_bytes is None:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None, None, val_err

        return obj_path, text_bytes, ""
    except Exception as exc:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None, None, f"freestanding -mx32 compile error: {exc}"


def _prepare_decompiled_for_sanitize(
    dec: DecompiledFunction,
    orig_name: str,
    safe_name: str,
) -> DecompiledFunction:
    """Normalize benign backend syntax (rizin `sym.` prefix, Hex-Rays `__cdecl`/`__intN`)
    without stripping any safety artifacts (`in_*`, `unaff_*`, `WARNING:`, or literals).
    """
    code = dec.c_code or ""
    if not code.strip():
        return dec

    # Normalize Hex-Rays __intN types to <stdint.h> types.
    for bits in ("64", "32", "16", "8"):
        code = re.sub(rf"\bunsigned\s+__int{bits}\b", f"uint{bits}_t", code)
        code = re.sub(rf"\bsigned\s+__int{bits}\b", f"int{bits}_t", code)
        code = re.sub(rf"\b__int{bits}\b", f"int{bits}_t", code)

    # Strip standard __cdecl keyword so GCC -mx32 accepts Hex-Rays prototypes.
    # Deliberately keep __stdcall/__fastcall/__usercall/__userpurge so invalid ABIs fail.
    code = re.sub(r"\b__cdecl\b\s*", "", code)

    # Strip rizin/Ghidra namespace prefix (e.g. `sym.plugin_compute(` -> `plugin_compute(`).
    code = re.sub(
        r"\b(?:sym|fcn|dbg)\." + re.escape(orig_name) + r"(\s*\()",
        f"{safe_name}\\1",
        code,
        count=1,
    )
    # Also handle stripped auto-generated names like `fcn.00001130(` or `sub_1130(`.
    if not re.search(r"\b" + re.escape(safe_name) + r"\s*\(", code):
        code = re.sub(
            r"\b(?:fcn\.[0-9a-fA-F]+|sub_[0-9a-fA-F]+)(\s*\()",
            f"{safe_name}\\1",
            code,
            count=1,
        )
    elif orig_name != safe_name:
        code = re.sub(
            r"\b" + re.escape(orig_name) + r"(\s*\()",
            f"{safe_name}\\1",
            code,
            count=1,
        )

    return DecompiledFunction(
        name=safe_name,
        va=dec.va,
        c_code=code,
        prototype=dec.prototype,
        backend=dec.backend,
        warnings=list(dec.warnings),
    )


def _verify_boundary_stub_link(
    stub_asm: str,
    boosted_obj_path: str,
    entry_label: str,
    pure_callees: List[str],
) -> Optional[str]:
    """Assemble boundary stub with `as --x32` and verify linkage via `ld -m elf32_x86_64`."""
    as_bin = shutil.which("as")
    ld_bin = shutil.which("ld")
    if not as_bin or not ld_bin:
        return "required binutils (as/ld) not found in PATH"

    with tempfile.TemporaryDirectory(prefix="nm_l3_link_") as td:
        stub_s = os.path.join(td, "stub.s")
        stub_o = os.path.join(td, "stub.o")
        out_bin = os.path.join(td, "linked.out")

        with open(stub_s, "w", encoding="utf-8") as fh:
            fh.write(stub_asm)

        p_as = subprocess.run(
            [as_bin, "--x32", stub_s, "-o", stub_o],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if p_as.returncode != 0 or not os.path.isfile(stub_o):
            err = (p_as.stderr or p_as.stdout or f"as exited {p_as.returncode}").strip()
            return f"as --x32 failed: {err[:400]}"

        if pure_callees:
            # Relocatable link verifies stub <-> boosted .o symbol resolution while
            # leaving permitted pure libm imports for final binary linking.
            cmd_ld = [ld_bin, "-m", "elf32_x86_64", "-r", stub_o, boosted_obj_path, "-o", out_bin]
        else:
            cmd_ld = [
                ld_bin,
                "-m",
                "elf32_x86_64",
                "-e",
                entry_label,
                stub_o,
                boosted_obj_path,
                "-o",
                out_bin,
            ]

        p_ld = subprocess.run(
            cmd_ld,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if p_ld.returncode != 0 or not os.path.isfile(out_bin):
            err = (p_ld.stderr or p_ld.stdout or f"ld exited {p_ld.returncode}").strip()
            return f"ld -m elf32_x86_64 failed: {err[:400]}"

    return None


def run_level3_boost(
    elf_path: str,
    functions: List[Tuple[int, str, bytes]],
    *,
    seg_ranges: List[Tuple[int, int]],
    known_imports: Dict[int, Tuple[str, str, str]],
    resolved_jump_tables: Optional[Set[int]] = None,
    decompiler_choice: str = "auto",
    trials: int = 4096,
    l2a_ref_asm_map: Optional[Dict[int, str]] = None,
    verbose: bool = False,
) -> Level3BoostReport:
    """Run the gated Level 3A boost pipeline with guaranteed Level 2A fallback on any failure."""
    outcomes: List[FunctionBoostOutcome] = []
    boosted_map: Dict[int, BoostedFunction] = {}
    gate_passed_items: List[Tuple[int, str, bytes, GateVerdict]] = []

    # Stage 1: Static purity gates (per function)
    for va, func_name, func_bytes in functions:
        if _is_pic_thunk(func_name, func_bytes):
            reason = "gates: PIC thunk helper (fallback to Level 2A)"
            outcomes.append(
                FunctionBoostOutcome(
                    name=func_name,
                    va=va,
                    boosted=False,
                    stage_failed="gate",
                    reason=reason,
                    gate_reasons=[reason],
                )
            )
            continue

        verdict = evaluate_purity_gate(
            func_name,
            va,
            func_bytes,
            seg_ranges=seg_ranges,
            known_imports=known_imports,
            resolved_jump_tables=resolved_jump_tables,
        )
        if not verdict.passed:
            reasons = list(verdict.reasons) or ["purity gate rejected"]
            outcomes.append(
                FunctionBoostOutcome(
                    name=func_name,
                    va=va,
                    boosted=False,
                    stage_failed="gate",
                    reason="; ".join(reasons),
                    gate_reasons=reasons,
                )
            )
            continue

        gate_passed_items.append((va, func_name, func_bytes, verdict))

    if not gate_passed_items:
        backend_label = decompiler_choice
        try:
            backend_obj = select_backend(decompiler_choice, require_headless=True)
            backend_label = backend_obj.name
        except Exception:
            pass
        return Level3BoostReport(
            backend=backend_label,
            total_functions=len(functions),
            gate_passed=0,
            boosted_count=0,
            outcomes=outcomes,
            boosted_map=boosted_map,
        )

    # Batch decompile ONLY gate-passing candidates in a single backend invocation.
    backend_name = decompiler_choice
    decompiled_by_va: Dict[int, DecompiledFunction] = {}
    batch_error: Optional[str] = None
    try:
        backend = select_backend(decompiler_choice, require_headless=True)
        backend_name = backend.name
        candidates = [(va, name) for va, name, _bytes, _verdict in gate_passed_items]
        if verbose:
            print(f"[level3:boost] Batch decompiling {len(candidates)} candidate(s) with {backend_name}")
        decompiled_by_va = backend.decompile_batch(elf_path, candidates)
    except Exception as exc:
        batch_error = f"backend decompilation failed: {exc}"

    if batch_error is not None:
        for va, func_name, _func_bytes, _verdict in gate_passed_items:
            outcomes.append(
                FunctionBoostOutcome(
                    name=func_name,
                    va=va,
                    boosted=False,
                    stage_failed="decompile",
                    reason=batch_error,
                    gate_reasons=[],
                )
            )
        return Level3BoostReport(
            backend=backend_name,
            total_functions=len(functions),
            gate_passed=len(gate_passed_items),
            boosted_count=0,
            outcomes=outcomes,
            boosted_map=boosted_map,
        )

    l2a_map = l2a_ref_asm_map or {}

    # Stages 2-5 per gate-passing function (never raise; always record outcome)
    for va, func_name, func_bytes, verdict in gate_passed_items:
        safe_name = _safe_c_name(func_name, va)
        dec = decompiled_by_va.get(va)
        if dec is None or not (dec.c_code or "").strip():
            warn_msg = "; ".join(dec.warnings) if dec and dec.warnings else "empty decompiler output"
            outcomes.append(
                FunctionBoostOutcome(
                    name=func_name,
                    va=va,
                    boosted=False,
                    stage_failed="decompile",
                    reason=warn_msg,
                    gate_reasons=[],
                )
            )
            continue

        # Stage 2: Sanitize
        prepared = _prepare_decompiled_for_sanitize(dec, func_name, safe_name)
        expected_sig = FunctionSignature(
            return_type=verdict.ret_type,
            param_types=list(verdict.arg_types),
        )
        san = sanitize_function(
            prepared,
            func_name=safe_name,
            expected_signature=expected_sig,
            segment_ranges=seg_ranges,
        )
        if not san.ok or not san.c_code:
            outcomes.append(
                FunctionBoostOutcome(
                    name=func_name,
                    va=va,
                    boosted=False,
                    stage_failed="sanitize",
                    reason=san.reason or "sanitization rejected",
                    gate_reasons=[],
                )
            )
            continue

        # Stage 3: Compile freestanding -mx32
        obj_path, text_bytes, comp_err = compile_freestanding_x32(
            san.c_code,
            safe_name,
            allowed_pure_externs=set(verdict.pure_callees),
        )
        if obj_path is None or text_bytes is None:
            outcomes.append(
                FunctionBoostOutcome(
                    name=func_name,
                    va=va,
                    boosted=False,
                    stage_failed="compile",
                    reason=comp_err or "compilation failed",
                    gate_reasons=[],
                )
            )
            continue

        try:
            with open(obj_path, "rb") as fh:
                obj_bytes = fh.read()

            # Stage 4: Two-tier differential verification (fail-closed)
            dv = verify_boosted_function(
                safe_name,
                va,
                func_bytes,
                obj_path,
                verdict.ret_type,
                list(verdict.arg_types),
                l2a_ref_asm=l2a_map.get(va),
                trials=trials,
            )
            if not dv.passed:
                reason = dv.reason or "differential verification failed"
                if dv.first_mismatch:
                    reason = f"{reason}: {dv.first_mismatch}"
                outcomes.append(
                    FunctionBoostOutcome(
                        name=func_name,
                        va=va,
                        boosted=False,
                        stage_failed="diffverify",
                        reason=reason,
                        gate_reasons=[],
                    )
                )
                continue

            # Stage 5: Boundary stub generation + as --x32 / ld -m elf32_x86_64 verification
            try:
                stub_asm = generate_cdecl_to_x32_boundary_stub(
                    safe_name, va, verdict.ret_type, list(verdict.arg_types)
                )
            except Exception as exc:
                outcomes.append(
                    FunctionBoostOutcome(
                        name=func_name,
                        va=va,
                        boosted=False,
                        stage_failed="stub",
                        reason=f"boundary stub generation failed: {exc}",
                        gate_reasons=[],
                    )
                )
                continue

            link_err = _verify_boundary_stub_link(
                stub_asm,
                obj_path,
                f"__nm_l3_entry_{va:x}",
                list(verdict.pure_callees),
            )
            if link_err is not None:
                outcomes.append(
                    FunctionBoostOutcome(
                        name=func_name,
                        va=va,
                        boosted=False,
                        stage_failed="stub",
                        reason=link_err,
                        gate_reasons=[],
                    )
                )
                continue

            bf = BoostedFunction(
                name=func_name,
                va=va,
                size=len(func_bytes),
                ret_type=verdict.ret_type,
                arg_types=list(verdict.arg_types),
                backend=dec.backend or backend_name,
                c_source=san.c_code,
                obj_bytes=obj_bytes,
                text_bytes=text_bytes,
                boundary_stub_asm=stub_asm,
                diff_mode=dv.mode,
                trials_run=dv.trials_run,
            )
            boosted_map[va] = bf
            outcomes.append(
                FunctionBoostOutcome(
                    name=func_name,
                    va=va,
                    boosted=True,
                    stage_failed=None,
                    reason="verified",
                    gate_reasons=[],
                    boosted_func=bf,
                )
            )
        finally:
            shutil.rmtree(os.path.dirname(obj_path), ignore_errors=True)

    return Level3BoostReport(
        backend=backend_name,
        total_functions=len(functions),
        gate_passed=len(gate_passed_items),
        boosted_count=len(boosted_map),
        outcomes=outcomes,
        boosted_map=boosted_map,
    )
