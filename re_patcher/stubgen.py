"""Specialized AT&T x32 GAS Marshaling Stub Generator for Imported Symbols.

Generates 12-15 instruction specialized AT&T x32 GAS marshaling stubs per imported
symbol directly from LookupSymEntry / ParsedSig DSL, eliminating the generic ~80-120
instruction interpreter hop (newminhh_call_x32_cdecl -> newminhh_call_x32_sig).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any

try:
    from . import PatcherError
except ImportError:
    try:
        from re_patcher import PatcherError  # type: ignore[no-redef]
    except ImportError:
        class PatcherError(Exception):  # type: ignore[no-redef]
            def __init__(self, message: str, exit_code: int = 1) -> None:
                super().__init__(message)
                self.message = message
                self.exit_code = exit_code

try:
    from .lookup import (
        LookupSymEntry,
        LookupTable,
        ParsedSig,
        parse_sig,
        NM_RET_VOID,
        NM_RET_I32,
        NM_RET_U32,
        NM_RET_PTR,
        NM_RET_I64,
        NM_RET_U64,
        NM_RET_F32,
        NM_RET_F64,
        NM_ARG_NONE,
        NM_ARG_I32,
        NM_ARG_U32,
        NM_ARG_PTR,
        NM_ARG_I64,
        NM_ARG_U64,
        NM_ARG_F32,
        NM_ARG_F64,
        NM_SIG_EXT_SRET,
        NM_SIG_EXT_CALLEE_POP4,
        NM_SYM_FLAG_DATA,
        NM_SYM_FLAG_VARIADIC,
        NM_SYM_FLAG_CDECL_DIRECT,
        NM_SYM_FLAG_WEAK,
    )
except ImportError:
    # Standalone constants fallback
    NM_RET_VOID = 0
    NM_RET_I32 = 1
    NM_RET_U32 = 2
    NM_RET_PTR = 3
    NM_RET_I64 = 4
    NM_RET_U64 = 5
    NM_RET_F32 = 6
    NM_RET_F64 = 7

    NM_ARG_NONE = 0
    NM_ARG_I32 = 1
    NM_ARG_U32 = 2
    NM_ARG_PTR = 3
    NM_ARG_I64 = 4
    NM_ARG_U64 = 5
    NM_ARG_F32 = 6
    NM_ARG_F64 = 7

    NM_SIG_EXT_SRET = 1 << 0
    NM_SIG_EXT_CALLEE_POP4 = 1 << 1

    NM_SYM_FLAG_DATA = 1 << 0
    NM_SYM_FLAG_VARIADIC = 1 << 1
    NM_SYM_FLAG_CDECL_DIRECT = 1 << 2
    NM_SYM_FLAG_WEAK = 1 << 8

# x86_64 / x32 SysV ABI register pools
INT_REGS_32 = ("%edi", "%esi", "%edx", "%ecx", "%r8d", "%r9d")
INT_REGS_64 = ("%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9")
FP_REGS = ("%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7")


@dataclass
class StubGenResult:
    """Result of generating specialized marshaling stubs."""

    stubs_asm: str
    stub_classes: Dict[int, str]  # idx -> stub class name
    stubs_by_index: Dict[int, List[str]]


def classify_stub(
    entry: Optional[LookupSymEntry] = None, sig_str: Optional[str] = None
) -> str:
    """Classify stub generation strategy into one of:
      - PATCHED_CDECL: 2-instruction tail jump
      - CDECL_DIRECT: direct cdecl bridge (variadic or explicit cdecl_direct)
      - NATIVE_TYPED: specialized 12-15 instruction SysV register marshaling
      - GENERIC: fallback to newminhh_call_x32_cdecl interpreter
    """
    if entry is not None:
        if entry.cls == "PATCHED_CDECL":
            return "PATCHED_CDECL"
        is_cdecl_direct = bool(
            (entry.flags & NM_SYM_FLAG_CDECL_DIRECT)
            or ("CDECL_DIRECT" in (entry.raw_flags or ""))
            or getattr(entry, "is_cdecl_direct", False)
            or entry.cls == "COMPAT_CDECL"
        )
        if is_cdecl_direct:
            return "CDECL_DIRECT"

        parsed = entry.parsed_sig
        if parsed is None and entry.sig_str:
            try:
                parsed = parse_sig(entry.sig_str)
            except Exception:
                parsed = None

        if parsed is not None:
            # Check argument registers capacity: <= 6 int/ptr/i64, <= 8 float/double
            is_sret = bool(parsed.abi_ext & NM_SIG_EXT_SRET or parsed.sret_size > 0)
            sret_count = 1 if (is_sret and parsed.sret_size > 16) else 0
            int_args = sret_count + sum(
                1 for a in parsed.arg_kinds if a in (NM_ARG_I32, NM_ARG_U32, NM_ARG_PTR, NM_ARG_I64, NM_ARG_U64)
            )
            fp_args = sum(1 for a in parsed.arg_kinds if a in (NM_ARG_F32, NM_ARG_F64))
            if int_args <= 6 and fp_args <= 8:
                return "NATIVE_TYPED"
            return "GENERIC"

        if entry.is_untyped or entry.cls == "UNTYPED":
            return "GENERIC"

    if sig_str:
        if sig_str == "PATCHED_CDECL":
            return "PATCHED_CDECL"
        if sig_str == "CDECL_DIRECT":
            return "CDECL_DIRECT"
        try:
            parsed = parse_sig(sig_str)
        except Exception:
            parsed = None
        if parsed is not None:
            is_sret = bool(parsed.abi_ext & NM_SIG_EXT_SRET or parsed.sret_size > 0)
            sret_count = 1 if (is_sret and parsed.sret_size > 16) else 0
            int_args = sret_count + sum(
                1 for a in parsed.arg_kinds if a in (NM_ARG_I32, NM_ARG_U32, NM_ARG_PTR, NM_ARG_I64, NM_ARG_U64)
            )
            fp_args = sum(1 for a in parsed.arg_kinds if a in (NM_ARG_F32, NM_ARG_F64))
            if int_args <= 6 and fp_args <= 8:
                return "NATIVE_TYPED"

    return "GENERIC"


def generate_specialized_stub(
    idx: int,
    sym_name: str,
    entry: Optional[LookupSymEntry] = None,
    sig_str: Optional[str] = None,
) -> List[str]:
    """Generate specialized AT&T x32 GAS assembly lines for __nm_plt_{idx}.

    Exact Entry Contract (matching trampolines.py __nm_plt_*):
      - On entry: 0(%rsp) = 4-byte guest return VA, 4(%rsp) = arg0, 8(%rsp) = arg1...
      - Preserves i386 callee-saved %rbx, %rbp, %rsi, %rdi
        (only %rsi and %rdi saved/restored across SysV x32 call).
      - Aligns %rsp to 16 bytes before call *%r11.
    """
    cls = classify_stub(entry, sig_str)
    lines: List[str] = [
        f".globl __nm_plt_{idx}",
        f"__nm_plt_{idx}:",
    ]

    # -------------------------------------------------------------------------
    # 1. PATCHED_CDECL: 2-instruction tail jump
    # -------------------------------------------------------------------------
    if cls == "PATCHED_CDECL":
        lines.append(f"    movl sym_got_plt_{idx}(%rip), %r11d")
        lines.append("    jmp *%r11")
        return lines

    # -------------------------------------------------------------------------
    # 2. CDECL_DIRECT / VARIADIC
    # -------------------------------------------------------------------------
    if cls == "CDECL_DIRECT":
        lines.append("    movq %rsp, %r10")
        lines.append("    andq $-16, %rsp")
        lines.append("    subq $32, %rsp")
        lines.append("    movq %r10, 0(%rsp)")
        lines.append("    movq %rsi, 8(%rsp)")
        lines.append("    movq %rdi, 16(%rsp)")
        lines.append(f"    movl sym_got_plt_{idx}(%rip), %r11d")
        lines.append("    testl %r11d, %r11d")
        lines.append(f"    jz .Lstub_skip_{idx}")
        lines.append("    leaq 4(%r10), %rdi")
        lines.append("    call *%r11")
        lines.append(f".Lstub_skip_{idx}:")
        lines.append("    movq 16(%rsp), %rdi")
        lines.append("    movq 8(%rsp), %rsi")
        lines.append("    movq 0(%rsp), %rsp")
        lines.append("    movl (%rsp), %r11d")
        is_callee_pop = bool(entry and entry.parsed_sig and (entry.parsed_sig.abi_ext & NM_SIG_EXT_CALLEE_POP4))
        if is_callee_pop:
            lines.append("    leaq 8(%rsp), %rsp")
        else:
            lines.append("    leaq 4(%rsp), %rsp")
        lines.append("    jmp *%r11")
        return lines

    # -------------------------------------------------------------------------
    # 3. NATIVE_TYPED: Specialized 12-15 instruction SysV register marshaling
    # -------------------------------------------------------------------------
    parsed: Optional[ParsedSig] = None
    if entry and entry.parsed_sig:
        parsed = entry.parsed_sig
    elif sig_str:
        try:
            parsed = parse_sig(sig_str)
        except Exception:
            parsed = None

    if cls == "NATIVE_TYPED" and parsed is not None:
        is_sret = bool(parsed.abi_ext & NM_SIG_EXT_SRET or parsed.sret_size > 0)
        sret_size = parsed.sret_size if is_sret else 0

        # Frame size: 48 bytes if SRET (scratch slots 24(%rsp) and 32(%rsp)), else 32 bytes
        frame_sz = 48 if is_sret else 32

        lines.append("    movq %rsp, %r10")
        lines.append("    andq $-16, %rsp")
        lines.append(f"    subq ${frame_sz}, %rsp")
        lines.append("    movq %r10, 0(%rsp)")
        lines.append("    movq %rsi, 8(%rsp)")
        lines.append("    movq %rdi, 16(%rsp)")
        lines.append(f"    movl sym_got_plt_{idx}(%rip), %r11d")
        lines.append("    testl %r11d, %r11d")
        lines.append(f"    jz .Lstub_skip_{idx}")

        int_idx = 0
        fp_idx = 0

        if is_sret:
            if sret_size > 16:
                lines.append("    movl 4(%r10), %edi")
                int_idx = 1
            else:
                int_idx = 0
            offset = 8
        else:
            offset = 4

        for ak in parsed.arg_kinds:
            if ak == NM_ARG_I32:
                reg64 = INT_REGS_64[int_idx]
                lines.append(f"    movslq {offset}(%r10), {reg64}")
                int_idx += 1
                offset += 4
            elif ak in (NM_ARG_U32, NM_ARG_PTR):
                reg32 = INT_REGS_32[int_idx]
                lines.append(f"    movl {offset}(%r10), {reg32}")
                int_idx += 1
                offset += 4
            elif ak in (NM_ARG_I64, NM_ARG_U64):
                reg64 = INT_REGS_64[int_idx]
                lines.append(f"    movq {offset}(%r10), {reg64}")
                int_idx += 1
                offset += 8
            elif ak == NM_ARG_F32:
                xmm = FP_REGS[fp_idx]
                lines.append(f"    movss {offset}(%r10), {xmm}")
                fp_idx += 1
                offset += 4
            elif ak == NM_ARG_F64:
                xmm = FP_REGS[fp_idx]
                lines.append(f"    movsd {offset}(%r10), {xmm}")
                fp_idx += 1
                offset += 8

        lines.append("    call *%r11")

        # Return value fixup
        ret_k = parsed.ret_kind
        if is_sret:
            if sret_size <= 16:
                lines.append("    movq 0(%rsp), %r10")
                lines.append("    movq %rax, 24(%rsp)")
                lines.append("    movq %rdx, 32(%rsp)")
                lines.append("    movl 4(%r10), %eax")
                curr_n = sret_size
                curr_off = 0
                while curr_n >= 8:
                    lines.append(f"    movq {24 + curr_off}(%rsp), %rcx")
                    lines.append(f"    movq %rcx, {curr_off}(%rax)")
                    curr_n -= 8
                    curr_off += 8
                if curr_n >= 4:
                    lines.append(f"    movl {24 + curr_off}(%rsp), %ecx")
                    lines.append(f"    movl %ecx, {curr_off}(%rax)")
                    curr_n -= 4
                    curr_off += 4
                if curr_n >= 2:
                    lines.append(f"    movw {24 + curr_off}(%rsp), %cx")
                    lines.append(f"    movw %cx, {curr_off}(%rax)")
                    curr_n -= 2
                    curr_off += 2
                if curr_n >= 1:
                    lines.append(f"    movb {24 + curr_off}(%rsp), %cl")
                    lines.append(f"    movb %cl, {curr_off}(%rax)")
                    curr_n -= 1
                    curr_off += 1
            else:
                lines.append("    movq 0(%rsp), %r10")
                lines.append("    movl 4(%r10), %eax")
        elif ret_k in (NM_RET_VOID, NM_RET_I32, NM_RET_U32, NM_RET_PTR):
            pass  # %eax already holds 32-bit return value
        elif ret_k in (NM_RET_I64, NM_RET_U64):
            # SysV returned in %rax; i386 cdecl expects %edx:%eax
            lines.append("    movq %rax, %rdx")
            lines.append("    shrq $32, %rdx")
            lines.append("    movl %eax, %eax")
        elif ret_k == NM_RET_F32:
            # SysV returned in %xmm0; i386 cdecl expects ST(0) x87 top of stack
            lines.append("    movss %xmm0, 24(%rsp)")
            lines.append("    flds 24(%rsp)")
        elif ret_k == NM_RET_F64:
            # SysV returned in %xmm0; i386 cdecl expects ST(0) x87 top of stack
            lines.append("    movsd %xmm0, 24(%rsp)")
            lines.append("    fldl 24(%rsp)")

        # Epilogue
        lines.append(f".Lstub_skip_{idx}:")
        lines.append("    movq 16(%rsp), %rdi")
        lines.append("    movq 8(%rsp), %rsi")
        lines.append("    movq 0(%rsp), %rsp")
        lines.append("    movl (%rsp), %r11d")
        is_callee_pop = is_sret or bool(parsed.abi_ext & NM_SIG_EXT_CALLEE_POP4)
        if is_callee_pop:
            # SRET / CALLEE_POP4 pops 8 bytes (4B return VA + 4B sret pointer / arg)
            lines.append("    leaq 8(%rsp), %rsp")
        else:
            lines.append("    leaq 4(%rsp), %rsp")
        lines.append("    jmp *%r11")
        return lines

    # -------------------------------------------------------------------------
    # 4. GENERIC Fallback (newminhh_call_x32_cdecl interpreter bridge)
    # -------------------------------------------------------------------------
    lines.append("    movq %rsp, %r10")
    lines.append("    andq $-16, %rsp")
    lines.append("    subq $32, %rsp")
    lines.append("    movq %r10, 0(%rsp)")
    lines.append("    movq %rsi, 8(%rsp)")
    lines.append("    movq %rdi, 16(%rsp)")
    lines.append(f"    movl sym_got_plt_{idx}(%rip), %edi")
    lines.append("    testl %edi, %edi")
    lines.append(f"    jz .Lstub_skip_{idx}")
    lines.append(f"    leal __nm_sig_{idx}(%rip), %esi")
    lines.append("    leaq 4(%r10), %rdx")
    lines.append("    movl got_newminhh_call_x32_cdecl(%rip), %r11d")
    lines.append("    call *%r11")
    lines.append(f".Lstub_skip_{idx}:")
    lines.append("    movq 16(%rsp), %rdi")
    lines.append("    movq 8(%rsp), %rsi")
    lines.append("    movq 0(%rsp), %rsp")
    lines.append("    movl (%rsp), %r11d")
    is_callee_pop = bool(
        (entry and entry.parsed_sig and (entry.parsed_sig.abi_ext & NM_SIG_EXT_CALLEE_POP4))
        or (parsed and (parsed.abi_ext & NM_SIG_EXT_CALLEE_POP4))
    )
    if is_callee_pop:
        lines.append("    leaq 8(%rsp), %rsp")
    else:
        lines.append("    leaq 4(%rsp), %rsp")
    lines.append("    jmp *%r11")
    return lines


def generate_all_stubs(
    plt_entries: Sequence[Tuple[str, int, int]],
    lookup_table: Optional[LookupTable] = None,
    signatures: Optional[Dict[str, Any]] = None,
) -> StubGenResult:
    """Generate all specialized PLT stubs for imported symbols.

    plt_entries: list of (orig_sym, got_slot_va, slot_idx) tuples.
    Returns a StubGenResult containing complete assembly text.
    """
    all_lines: List[str] = [
        "/* Generated by newminhh specialized stub generator (Wave 3 Level 2A/2B) */",
        ".section .text.stubs, \"ax\"",
    ]

    stub_classes: Dict[int, str] = {}
    stubs_by_index: Dict[int, List[str]] = {}

    for idx, (orig_sym, _got_va, _slot_idx) in enumerate(plt_entries):
        entry: Optional[LookupSymEntry] = None
        if lookup_table is not None:
            entry = lookup_table.resolve_sym("*", orig_sym)

        sig_str: Optional[str] = None
        if entry is None and signatures and orig_sym in signatures:
            sig_info = signatures[orig_sym]
            if isinstance(sig_info, str):
                sig_str = sig_info
            elif isinstance(sig_info, (list, tuple)) and len(sig_info) >= 4:
                # COMPAT_SIGNATURES tuple: (ret, args, is_variadic, is_cdecl_direct)
                if sig_info[3]:  # is_cdecl_direct
                    sig_str = "CDECL_DIRECT"

        stub_cls = classify_stub(entry, sig_str)
        stub_classes[idx] = stub_cls

        stub_lines = generate_specialized_stub(idx, orig_sym, entry, sig_str)
        stubs_by_index[idx] = stub_lines
        all_lines.extend(stub_lines)
        all_lines.append("")

    return StubGenResult(
        stubs_asm="\n".join(all_lines),
        stub_classes=stub_classes,
        stubs_by_index=stubs_by_index,
    )
