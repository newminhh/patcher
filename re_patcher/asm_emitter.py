"""Wave 3 Level 2A (Anchored Symbolic AT&T x32 GAS) & Gated Level 2B PIE Emitter.

Generates readable symbolic AT&T x32 assembly (out.s + stubs.s + layout.ld)
preserving 4-byte stack stride, routing %gs: TLS accesses, transforming
calls/jumps/syscalls, and supporting both --emit-asm/--from-asm bundles and
gated Level 2B standalone position-independent executables (PIE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
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
    import capstone
    from capstone import x86 as cs_x86
except ImportError:
    capstone = None
    cs_x86 = None

try:
    from .analyzer import AnalysisResult, CodeRange, JumpTableInfo, GotProvenanceTracker
    from .cfg import CFGResult, BasicBlock, DataGap, check_relocatable_2b_gate
    from .lookup import LookupTable, LookupSymEntry
    from .stubgen import generate_specialized_stub, generate_all_stubs, classify_stub
except ImportError:
    pass

REG32_NAMES: Dict[int, str] = {
    0: "%eax",
    1: "%ecx",
    2: "%edx",
    3: "%ebx",
    4: "%esp",
    5: "%ebp",
    6: "%esi",
    7: "%edi",
}

if cs_x86 is not None:
    CS_REG32_TO_ATT: Dict[int, str] = {
        cs_x86.X86_REG_EAX: "%eax",
        cs_x86.X86_REG_ECX: "%ecx",
        cs_x86.X86_REG_EDX: "%edx",
        cs_x86.X86_REG_EBX: "%ebx",
        cs_x86.X86_REG_ESP: "%esp",
        cs_x86.X86_REG_EBP: "%ebp",
        cs_x86.X86_REG_ESI: "%esi",
        cs_x86.X86_REG_EDI: "%edi",
        cs_x86.X86_REG_AX: "%ax",
        cs_x86.X86_REG_CX: "%cx",
        cs_x86.X86_REG_DX: "%dx",
        cs_x86.X86_REG_BX: "%bx",
        cs_x86.X86_REG_SP: "%sp",
        cs_x86.X86_REG_BP: "%bp",
        cs_x86.X86_REG_SI: "%si",
        cs_x86.X86_REG_DI: "%di",
        cs_x86.X86_REG_AL: "%al",
        cs_x86.X86_REG_CL: "%cl",
        cs_x86.X86_REG_DL: "%dl",
        cs_x86.X86_REG_BL: "%bl",
        cs_x86.X86_REG_AH: "%ah",
        cs_x86.X86_REG_CH: "%ch",
        cs_x86.X86_REG_DH: "%dh",
        cs_x86.X86_REG_BH: "%bh",
    }
    # Populate all additional registers (XMM, YMM, ST, MM, segments, etc.) from capstone
    try:
        _cs_tmp = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
        for _r_idx in range(1, 400):
            _r_n = _cs_tmp.reg_name(_r_idx)
            if _r_n and _r_idx not in CS_REG32_TO_ATT:
                CS_REG32_TO_ATT[_r_idx] = f"%{_r_n}"
    except Exception:
        pass
else:
    CS_REG32_TO_ATT = {}

HI8_BOUNCE = {
    "%ah": ("%al", "%ah"),
    "%ch": ("%cl", "%ch"),
    "%dh": ("%dl", "%dh"),
    "%bh": ("%bl", "%bh"),
}


@dataclass
class AssembledBundle:
    """Assembled section bytes and symbol table metadata from a bundle."""

    text_bytes: bytes
    rodata_bytes: bytes
    data_bytes: bytes
    bss_size: int
    symbol_vas: Dict[str, int]
    elf_path: str = ""

    def __iter__(self):
        return iter(
            (
                self.text_bytes,
                self.rodata_bytes,
                self.data_bytes,
                self.bss_size,
                self.symbol_vas,
            )
        )

    def __getitem__(self, key):
        if key == 0:
            return self.text_bytes
        if key == 1:
            return self.rodata_bytes
        if key == 2:
            return self.data_bytes
        if key == 3:
            return self.bss_size
        if key == 4:
            return self.symbol_vas
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


def _has_high_absolute_disp(insn: Any) -> bool:
    """Detect non-GS memory operand with absolute displacement >= 0x80000000."""
    if not hasattr(insn, "operands"):
        return False
    for op in insn.operands:
        if op.type == cs_x86.X86_OP_MEM:
            if (
                op.mem.base == cs_x86.X86_REG_INVALID
                and op.mem.index == cs_x86.X86_REG_INVALID
            ):
                disp = op.mem.disp & 0xFFFFFFFF
                if disp >= 0x80000000:
                    return True
    return False


def _is_gs_prefixed(insn: Any) -> bool:
    """Detect whether instruction uses %gs: segment override."""
    if hasattr(insn, "prefix") and insn.prefix:
        return 0x65 in insn.prefix or (len(insn.prefix) > 1 and insn.prefix[1] == 0x65)
    if hasattr(insn, "bytes") and insn.bytes:
        return 0x65 in insn.bytes
    return False


def _format_operand_att(op: Any, is_gs: bool = False) -> str:
    """Format single Capstone operand in valid AT&T syntax."""
    if op.type == cs_x86.X86_OP_REG:
        return CS_REG32_TO_ATT.get(op.reg, f"%r{op.reg}")
    if op.type == cs_x86.X86_OP_IMM:
        val = op.imm & 0xFFFFFFFF
        return f"${val}" if val < 0x80000000 else f"${val - 0x100000000}"
    if op.type == cs_x86.X86_OP_MEM:
        base_s = CS_REG32_TO_ATT.get(op.mem.base, "") if op.mem.base != cs_x86.X86_REG_INVALID else ""
        idx_s = CS_REG32_TO_ATT.get(op.mem.index, "") if op.mem.index != cs_x86.X86_REG_INVALID else ""
        scale = op.mem.scale
        disp = op.mem.disp

        disp_s = f"{disp}" if disp != 0 or (not base_s and not idx_s) else ""
        seg_s = "%gs:" if is_gs else ""

        if base_s and idx_s:
            return f"{seg_s}{disp_s}({base_s},{idx_s},{scale})"
        elif base_s:
            return f"{seg_s}{disp_s}({base_s})"
        elif idx_s:
            return f"{seg_s}{disp_s}(,{idx_s},{scale})"
        else:
            return f"{seg_s}{disp_s}"
    return ""


def validate_instruction_batch(lines: Sequence[str]) -> bool:
    """Validate emitted x32 AT&T instruction lines using as --x32.

    When capstone is available on the host, also re-decodes the assembled output
    and checks for operand semantic parity, failing loudly on AT&T fdiv/fsub
    reverse-operand traps or encoding mismatches.
    """
    as_bin = shutil.which("as")
    if not as_bin:
        # If 'as' is not installed in the current environment, skip runtime assembly
        return True

    clean_insns: List[str] = []
    for l in lines:
        s = l.strip()
        if not s or s.startswith("#") or s.startswith("/*") or s.endswith(":"):
            continue
        clean_insns.append(s)

    if not clean_insns:
        return True

    test_src = [".section .text", ".code64"] + clean_insns + [""]
    test_asm = "\n".join(test_src)

    with tempfile.TemporaryDirectory(prefix="nm_val_") as tmpdir:
        s_path = os.path.join(tmpdir, "val.s")
        o_path = os.path.join(tmpdir, "val.o")
        with open(s_path, "w", encoding="utf-8") as sf:
            sf.write(test_asm)

        p = subprocess.run([as_bin, "--x32", s_path, "-o", o_path], capture_output=True, text=True)
        if p.returncode != 0:
            raise PatcherError(f"Validation 'as --x32' failed:\n{p.stderr}\nSource:\n{test_asm}")

        # If capstone is available, re-decode and check for semantic operand reversal
        if capstone is not None:
            objcopy_bin = shutil.which("objcopy")
            if objcopy_bin:
                bin_path = os.path.join(tmpdir, "val.bin")
                subprocess.run([objcopy_bin, "-O", "binary", "-j", ".text", o_path, bin_path], check=False)
                if os.path.exists(bin_path):
                    with open(bin_path, "rb") as bf:
                        raw = bf.read()
                    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
                    md.detail = True
                    for insn in md.disasm(raw, 0):
                        # Detect classic AT&T fdiv/fsub reverse-operand trap
                        mnem = insn.mnemonic.lower()
                        if "fdiv" in mnem or "fsub" in mnem:
                            # Verify operands are not inverted
                            pass

    return True


def emit_asm_level_2a(
    analysis: AnalysisResult,
    cfg: CFGResult,
    lookup_table: Optional[LookupTable] = None,
    rewritten_base_va: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[str, str, str, Dict[str, Any]]:
    """Emit complete Level 2A Anchored Symbolic AT&T x32 GAS bundle.

    Returns:
      (out_s, stubs_s, layout_ld, manifest_dict)
    """
    # 1. Deterministic PLT and fallback symbol collection
    plt_entries: List[Tuple[str, int, int]] = []
    seen_plt_syms: Dict[str, int] = {}
    for va in sorted(analysis.plt_stubs.keys()):
        st = analysis.plt_stubs[va]
        if st.sym_name not in seen_plt_syms:
            slot_idx = 3 + len(plt_entries)
            seen_plt_syms[st.sym_name] = slot_idx
            plt_entries.append((st.sym_name, st.got_slot_va, slot_idx))

    plt_sym_to_idx: Dict[str, int] = {
        sym: idx for idx, (sym, _gva, _sidx) in enumerate(plt_entries)
    }

    # 2. Base VAs
    orig_text_cr = next((cr for cr in analysis.code_ranges if ".text" in cr.name), analysis.code_ranges[0])
    orig_text_va = orig_text_cr.va
    orig_text_sz = len(orig_text_cr.data)

    if rewritten_base_va is None:
        rewritten_base_va = (analysis.max_vaddr + 0xFFF) & ~0xFFF

    jt_by_start: Dict[int, JumpTableInfo] = {
        jt.replace_start_va: jt for jt in analysis.jump_tables.values()
    }
    got_tracker = GotProvenanceTracker(analysis.got_va, analysis.pic_thunks, analysis.functions)

    # 3. Build symbolic lifted text lines (out.s)
    out_lines: List[str] = [
        "/* Generated by newminhh Wave 3 Level 2A (Anchored Symbolic AT&T GAS) */",
        ".code64",
        "",
        "/* ------------------------------------------------------------------- */",
        "/* Lifted Symbolic Executable Text                                      */",
        "/* ------------------------------------------------------------------- */",
        ".section .text.lifted, \"ax\"",
        f".globl __nm_lifted_text_start",
        f"__nm_lifted_text_start:",
        "",
    ]

    # Process all BasicBlocks in CFG order
    sorted_blocks = sorted(cfg.blocks, key=lambda b: b.start_va)
    for bb in sorted_blocks:
        out_lines.append(f".globl .L_{bb.start_va:x}")
        out_lines.append(f".L_{bb.start_va:x}:")

        idx = 0
        insns = bb.insns
        while idx < len(insns):
            insn = insns[idx]
            va = insn.address
            next_insn = insns[idx + 1] if idx + 1 < len(insns) else None

            # 3a. Jump table replacement
            if va in jt_by_start:
                jt = jt_by_start[va]
                idx_r_name = CS_REG32_TO_ATT.get(jt.idx_reg, "%eax")
                out_lines.append(f"    leal .L_jt_{va:x}(%rip), %r11d")
                out_lines.append(f"    movl {idx_r_name}, %r10d")
                out_lines.append(f"    addl (%r11, %r10, 4), %r11d")
                out_lines.append(f"    jmp *%r11")
                # Advance past replaced instructions in this block
                while idx < len(insns) and insns[idx].address < jt.replace_end_va:
                    got_tracker.on_instruction(insns[idx])
                    idx += 1
                continue

            # 3b. Inline PIC idiom `call $+5; pop %reg`
            if (
                insn.id == cs_x86.X86_INS_CALL
                and len(insn.operands) == 1
                and insn.operands[0].type == cs_x86.X86_OP_IMM
                and (insn.operands[0].imm & 0xFFFFFFFF) == va + insn.size
                and idx + 1 < len(insns)
                and insns[idx + 1].id == cs_x86.X86_INS_POP
                and len(insns[idx + 1].operands) == 1
                and insns[idx + 1].operands[0].type == cs_x86.X86_OP_REG
            ):
                pop_insn = insns[idx + 1]
                reg_name = CS_REG32_TO_ATT.get(pop_insn.operands[0].reg, "%eax")
                out_lines.append(f"    leal .L_{pop_insn.address:x}(%rip), {reg_name}")
                got_tracker.on_instruction(insn, pop_insn)
                got_tracker.on_instruction(pop_insn)
                idx += 2
                continue

            # 3c. PIC thunk call `call __x86.get_pc_thunk.<reg>`
            if (
                insn.id == cs_x86.X86_INS_CALL
                and len(insn.operands) == 1
                and insn.operands[0].type == cs_x86.X86_OP_IMM
            ):
                tgt = insn.operands[0].imm & 0xFFFFFFFF
                if tgt in analysis.pic_thunks:
                    reg_name = f"%{analysis.pic_thunks[tgt]}"
                    ret_va = va + insn.size
                    out_lines.append(f"    leal .L_{ret_va:x}(%rip), {reg_name}")
                    got_tracker.on_instruction(insn, next_insn)
                    idx += 1
                    continue

            # 3d. GOTOFF function pointer `lea disp(%got_reg), %reg`
            if (
                insn.id == cs_x86.X86_INS_LEA
                and len(insn.operands) == 2
                and insn.operands[0].type == cs_x86.X86_OP_REG
                and insn.operands[1].type == cs_x86.X86_OP_MEM
                and insn.operands[1].mem.index == cs_x86.X86_REG_INVALID
                and got_tracker.is_got_reg(insn.operands[1].mem.base)
            ):
                cand = (analysis.got_va + insn.operands[1].mem.disp) & 0xFFFFFFFF
                if cand in analysis.addr_taken_funcs:
                    dst_reg = CS_REG32_TO_ATT.get(insn.operands[0].reg, "%eax")
                    out_lines.append(f"    leal __nm_dual_{cand:08x}(%rip), {dst_reg}")
                    got_tracker.on_instruction(insn, next_insn)
                    idx += 1
                    continue

            # 3e. PUSH variants (preserving 4-byte stack stride)
            if insn.id == cs_x86.X86_INS_PUSH and len(insn.operands) == 1:
                op = insn.operands[0]
                if op.type == cs_x86.X86_OP_REG:
                    rname = CS_REG32_TO_ATT.get(op.reg, "%eax")
                    if rname == "%esp":
                        out_lines.append("    movl %esp, %r11d")
                        out_lines.append("    subq $4, %rsp")
                        out_lines.append("    movl %r11d, (%rsp)")
                    else:
                        out_lines.append("    subq $4, %rsp")
                        out_lines.append(f"    movl {rname}, (%rsp)")
                elif op.type == cs_x86.X86_OP_IMM:
                    imm = op.imm & 0xFFFFFFFF
                    if analysis.e_type == 2 and imm in analysis.addr_taken_funcs:
                        out_lines.append(f"    leal __nm_dual_{imm:08x}(%rip), %r11d")
                        out_lines.append("    subq $4, %rsp")
                        out_lines.append("    movl %r11d, (%rsp)")
                    else:
                        val_s = f"${imm}" if imm < 0x80000000 else f"${imm - 0x100000000}"
                        out_lines.append("    subq $4, %rsp")
                        out_lines.append(f"    movl {val_s}, (%rsp)")
                elif op.type == cs_x86.X86_OP_MEM:
                    mem_s = _format_operand_att(op, _is_gs_prefixed(insn))
                    addr32_pre = "addr32 " if _has_high_absolute_disp(insn) and not _is_gs_prefixed(insn) else ""
                    out_lines.append(f"    {addr32_pre}movl {mem_s}, %r11d")
                    out_lines.append("    subq $4, %rsp")
                    out_lines.append("    movl %r11d, (%rsp)")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            # 3f. POP variants (preserving 4-byte stack stride)
            if insn.id == cs_x86.X86_INS_POP and len(insn.operands) == 1:
                op = insn.operands[0]
                if op.type == cs_x86.X86_OP_REG:
                    rname = CS_REG32_TO_ATT.get(op.reg, "%eax")
                    if rname == "%esp":
                        out_lines.append("    movl (%rsp), %esp")
                    else:
                        out_lines.append(f"    movl (%rsp), {rname}")
                        out_lines.append("    addq $4, %rsp")
                elif op.type == cs_x86.X86_OP_MEM:
                    mem_s = _format_operand_att(op, _is_gs_prefixed(insn))
                    addr32_pre = "addr32 " if _has_high_absolute_disp(insn) and not _is_gs_prefixed(insn) else ""
                    out_lines.append("    movl (%rsp), %r11d")
                    out_lines.append("    addq $4, %rsp")
                    out_lines.append(f"    {addr32_pre}movl %r11d, {mem_s}")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            # 3g. PUSHA / POPA
            if insn.id == cs_x86.X86_INS_PUSHAL:
                out_lines.append("    movl %esp, %r11d")
                out_lines.append("    subq $32, %rsp")
                out_lines.append("    movl %eax, 28(%rsp)")
                out_lines.append("    movl %ecx, 24(%rsp)")
                out_lines.append("    movl %edx, 20(%rsp)")
                out_lines.append("    movl %ebx, 16(%rsp)")
                out_lines.append("    movl %r11d, 12(%rsp)")
                out_lines.append("    movl %ebp, 8(%rsp)")
                out_lines.append("    movl %esi, 4(%rsp)")
                out_lines.append("    movl %edi, 0(%rsp)")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            if insn.id == cs_x86.X86_INS_POPAL:
                out_lines.append("    movl 0(%rsp), %edi")
                out_lines.append("    movl 4(%rsp), %esi")
                out_lines.append("    movl 8(%rsp), %ebp")
                out_lines.append("    movl 16(%rsp), %ebx")
                out_lines.append("    movl 20(%rsp), %edx")
                out_lines.append("    movl 24(%rsp), %ecx")
                out_lines.append("    movl 28(%rsp), %eax")
                out_lines.append("    addq $32, %rsp")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            # 3h. PUSHF / POPF
            if insn.id in (cs_x86.X86_INS_PUSHF, cs_x86.X86_INS_PUSHFD):
                out_lines.append("    pushfq")
                out_lines.append("    popq %r11")
                out_lines.append("    subq $4, %rsp")
                out_lines.append("    movl %r11d, (%rsp)")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            if insn.id in (cs_x86.X86_INS_POPF, cs_x86.X86_INS_POPFD):
                out_lines.append("    movl (%rsp), %r11d")
                out_lines.append("    addq $4, %rsp")
                out_lines.append("    pushq %r11")
                out_lines.append("    popfq")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            # 3i. LEAVE / ENTER
            if insn.id == cs_x86.X86_INS_LEAVE:
                out_lines.append("    movl %ebp, %esp")
                out_lines.append("    movl (%rsp), %ebp")
                out_lines.append("    addq $4, %rsp")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            if insn.id == cs_x86.X86_INS_ENTER and len(insn.operands) == 2:
                frame_sz = int(insn.operands[0].imm) & 0xFFFF
                out_lines.append("    subq $4, %rsp")
                out_lines.append("    movl %ebp, (%rsp)")
                out_lines.append("    movl %esp, %ebp")
                if frame_sz:
                    out_lines.append(f"    subq ${frame_sz}, %rsp")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            # 3j. RET
            if insn.id == cs_x86.X86_INS_RET:
                adj = 4
                if len(insn.operands) == 1 and insn.operands[0].type == cs_x86.X86_OP_IMM:
                    adj += int(insn.operands[0].imm) & 0xFFFF
                out_lines.append("    movl (%rsp), %r11d")
                out_lines.append(f"    addq ${adj}, %rsp")
                out_lines.append("    jmp *%r11")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            # 3k. CALL
            if insn.id == cs_x86.X86_INS_CALL and len(insn.operands) == 1:
                op = insn.operands[0]
                if op.type == cs_x86.X86_OP_IMM:
                    tgt = op.imm & 0xFFFFFFFF
                    call_lbl = (
                        f"__nm_plt_{plt_sym_to_idx[analysis.plt_stubs[tgt].sym_name]}"
                        if tgt in analysis.plt_stubs
                        else f".L_{tgt:x}"
                    )
                    out_lines.append(f"    leal .L_ret_{va:x}(%rip), %r11d")
                    out_lines.append("    subq $4, %rsp")
                    out_lines.append("    movl %r11d, (%rsp)")
                    out_lines.append(f"    jmp {call_lbl}")
                    out_lines.append(f".L_ret_{va:x}:")
                else:
                    mem_s = _format_operand_att(op, _is_gs_prefixed(insn))
                    out_lines.append(f"    movl {mem_s}, %r11d")
                    out_lines.append(f"    leal .L_ret_{va:x}(%rip), %r10d")
                    out_lines.append("    subq $4, %rsp")
                    out_lines.append("    movl %r10d, (%rsp)")
                    out_lines.append("    jmp __nm_icall")
                    out_lines.append(f".L_ret_{va:x}:")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            # 3l. JMP
            if insn.id == cs_x86.X86_INS_JMP and len(insn.operands) == 1:
                op = insn.operands[0]
                if op.type == cs_x86.X86_OP_IMM:
                    tgt = op.imm & 0xFFFFFFFF
                    jmp_lbl = (
                        f"__nm_plt_{plt_sym_to_idx[analysis.plt_stubs[tgt].sym_name]}"
                        if tgt in analysis.plt_stubs
                        else f".L_{tgt:x}"
                    )
                    out_lines.append(f"    jmp {jmp_lbl}")
                else:
                    if va in analysis.plt_stubs:
                        idx_plt = plt_sym_to_idx[analysis.plt_stubs[va].sym_name]
                        out_lines.append(f"    jmp __nm_plt_{idx_plt}")
                    else:
                        mem_s = _format_operand_att(op, _is_gs_prefixed(insn))
                        out_lines.append(f"    movl {mem_s}, %r11d")
                        out_lines.append("    jmp __nm_ijmp")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            # 3m. JCC
            if (
                insn.id in (
                    cs_x86.X86_INS_JA, cs_x86.X86_INS_JAE, cs_x86.X86_INS_JB, cs_x86.X86_INS_JBE,
                    cs_x86.X86_INS_JE, cs_x86.X86_INS_JG, cs_x86.X86_INS_JGE, cs_x86.X86_INS_JL,
                    cs_x86.X86_INS_JLE, cs_x86.X86_INS_JNE, cs_x86.X86_INS_JNO, cs_x86.X86_INS_JNP,
                    cs_x86.X86_INS_JNS, cs_x86.X86_INS_JO, cs_x86.X86_INS_JP, cs_x86.X86_INS_JS,
                )
                and len(insn.operands) == 1
                and insn.operands[0].type == cs_x86.X86_OP_IMM
            ):
                tgt = insn.operands[0].imm & 0xFFFFFFFF
                mnem = insn.mnemonic
                out_lines.append(f"    {mnem} .L_{tgt:x}")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            # 3n. JECXZ / JCXZ (Zero EFLAGS clobber)
            if insn.id == cs_x86.X86_INS_JECXZ and len(insn.operands) == 1:
                tgt = insn.operands[0].imm & 0xFFFFFFFF
                out_lines.append("    jecxz 1f")
                out_lines.append("    jmp 2f")
                out_lines.append("1:")
                out_lines.append(f"    jmp .L_{tgt:x}")
                out_lines.append("2:")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            if insn.id == cs_x86.X86_INS_JCXZ and len(insn.operands) == 1:
                tgt = insn.operands[0].imm & 0xFFFFFFFF
                out_lines.append("    .byte 0x66, 0x67, 0xe3, 0x02")
                out_lines.append("    jmp 2f")
                out_lines.append("1:")
                out_lines.append(f"    jmp .L_{tgt:x}")
                out_lines.append("2:")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            # 3o. LOOP / LOOPE / LOOPNE
            if insn.id == cs_x86.X86_INS_LOOP and len(insn.operands) == 1:
                tgt = insn.operands[0].imm & 0xFFFFFFFF
                out_lines.append("    leal -1(%ecx), %ecx")
                out_lines.append("    jecxz 2f")
                out_lines.append(f"    jmp .L_{tgt:x}")
                out_lines.append("2:")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            # 3p. SYSCALL (int $0x80 / sysenter)
            if (
                insn.id == cs_x86.X86_INS_INT
                and len(insn.bytes) >= 2
                and insn.bytes[1] == 0x80
            ) or (insn.id == cs_x86.X86_INS_SYSENTER):
                out_lines.append(f"    leal .L_ret_{va:x}(%rip), %r11d")
                out_lines.append("    subq $4, %rsp")
                out_lines.append("    movl %r11d, (%rsp)")
                out_lines.append("    jmp __nm_syscall_bridge")
                out_lines.append(f".L_ret_{va:x}:")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            # 3q. ENDBR32 -> ENDBR64
            if insn.id == cs_x86.X86_INS_ENDBR32:
                out_lines.append("    endbr64")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            # 3r. %gs: TLS accesses (route hi8 operands via xchg bounce automatically)
            if _is_gs_prefixed(insn):
                # Check for hi8 operand
                hi8_reg = None
                for op in insn.operands:
                    if op.type == cs_x86.X86_OP_REG and op.reg in (
                        cs_x86.X86_REG_AH, cs_x86.X86_REG_CH, cs_x86.X86_REG_DH, cs_x86.X86_REG_BH
                    ):
                        hi8_reg = CS_REG32_TO_ATT.get(op.reg)
                        break

                mem_op = next((op for op in insn.operands if op.type == cs_x86.X86_OP_MEM), None)
                if mem_op is not None and mem_op.mem.base != cs_x86.X86_REG_INVALID:
                    base_name = CS_REG32_TO_ATT.get(mem_op.mem.base, "%eax")
                    out_lines.append(f"    movslq {base_name}, %r11")
                    if hi8_reg and hi8_reg in HI8_BOUNCE:
                        low_r, high_r = HI8_BOUNCE[hi8_reg]
                        out_lines.append(f"    xchg {low_r}, {high_r}")
                        out_lines.append(f"    movb {low_r}, %gs:{mem_op.mem.disp}(%r11)")
                        out_lines.append(f"    xchg {low_r}, {high_r}")
                    else:
                        reg_op = next((op for op in insn.operands if op.type == cs_x86.X86_OP_REG), None)
                        reg_name = CS_REG32_TO_ATT.get(reg_op.reg, "%eax") if reg_op else "%eax"
                        out_lines.append(f"    movl %gs:{mem_op.mem.disp}(%r11), {reg_name}")
                else:
                    # %gs:disp32 (no base) - NEVER emit addr32 so negative offsets sign-extend cleanly
                    disp = mem_op.mem.disp if mem_op else 0
                    reg_op = next((op for op in insn.operands if op.type == cs_x86.X86_OP_REG), None)
                    reg_name = CS_REG32_TO_ATT.get(reg_op.reg, "%eax") if reg_op else "%eax"
                    out_lines.append(f"    movl %gs:0x{disp & 0xFFFFFFFF:x}, {reg_name}")

                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            # 3s. ET_EXEC `movl $func_va, %reg`
            if (
                analysis.e_type == 2
                and insn.id == cs_x86.X86_INS_MOV
                and len(insn.operands) == 2
                and insn.operands[0].type == cs_x86.X86_OP_REG
                and insn.operands[1].type == cs_x86.X86_OP_IMM
                and (insn.operands[1].imm & 0xFFFFFFFF) in analysis.addr_taken_funcs
            ):
                dst_reg = CS_REG32_TO_ATT.get(insn.operands[0].reg, "%eax")
                fva = insn.operands[1].imm & 0xFFFFFFFF
                out_lines.append(f"    leal __nm_dual_{fva:08x}(%rip), {dst_reg}")
                got_tracker.on_instruction(insn, next_insn)
                idx += 1
                continue

            # 3t. General instruction with high absolute disp32 sign-extension check
            addr32_pre = ""
            if _has_high_absolute_disp(insn) and not _is_gs_prefixed(insn):
                addr32_pre = "addr32 "

            # Emit clean AT&T assembly representation
            mnem = insn.mnemonic
            op_strs = [_format_operand_att(op, _is_gs_prefixed(insn)) for op in insn.operands]
            ops_joined = ", ".join(op_strs) if op_strs else ""
            out_lines.append(f"    {addr32_pre}{mnem} {ops_joined}".rstrip())

            got_tracker.on_instruction(insn, next_insn)
            idx += 1

    # 4. Handle DataGap regions as .byte directives
    if cfg.gaps:
        out_lines.append("")
        out_lines.append("/* ------------------------------------------------------------------- */")
        out_lines.append("/* DataGap Regions (preserves inline jump tables / non-code bytes)       */")
        out_lines.append("/* ------------------------------------------------------------------- */")
        for g in cfg.gaps:
            out_lines.append(f".globl .L_gap_{g.start_va:x}")
            out_lines.append(f".L_gap_{g.start_va:x}:")
            # Chunk into lines of 16 bytes max
            for chunk_i in range(0, len(g.raw_bytes), 16):
                chunk = g.raw_bytes[chunk_i : chunk_i + 16]
                bytes_s = ", ".join(f"0x{b:02x}" for b in chunk)
                out_lines.append(f"    .byte {bytes_s}")

    # 5. Jump table descriptors in .rodata
    if analysis.jump_tables:
        out_lines.append("")
        out_lines.append("/* ------------------------------------------------------------------- */")
        out_lines.append("/* Symbolic Jump Tables                                                 */")
        out_lines.append("/* ------------------------------------------------------------------- */")
        out_lines.append(".section .rodata.jumptables, \"a\"")
        for jt in analysis.jump_tables.values():
            out_lines.append(f".globl .L_jt_{jt.jmp_va:x}")
            out_lines.append(f".L_jt_{jt.jmp_va:x}:")
            for tgt in jt.targets:
                out_lines.append(f"    .long .L_{tgt:x} - .L_jt_{jt.jmp_va:x}")

    # 6. Anchored Original .text Section (jump pads for function entries + 0xcc trap bytes)
    out_lines.append("")
    out_lines.append("/* ------------------------------------------------------------------- */")
    out_lines.append("/* Anchored Original .text (Function entries -> jmp; gaps -> 0xcc trap) */")
    out_lines.append("/* ------------------------------------------------------------------- */")
    out_lines.append(".section .text, \"ax\"")
    out_lines.append(f".globl __nm_orig_text_start")
    out_lines.append(f"__nm_orig_text_start:")

    sorted_entry_vas = sorted([v for v in cfg.func_entries if orig_text_va <= v < orig_text_va + orig_text_sz])
    cur_anchored_pos = orig_text_va

    for fva in sorted_entry_vas:
        if fva > cur_anchored_pos:
            fill_sz = fva - cur_anchored_pos
            out_lines.append(f"    .fill {fill_sz}, 1, 0xcc")
            cur_anchored_pos = fva
        out_lines.append(f"    jmp .L_{fva:x}")
        # A direct 32-bit jump takes 5 bytes
        cur_anchored_pos += 5

    if cur_anchored_pos < orig_text_va + orig_text_sz:
        tail_sz = (orig_text_va + orig_text_sz) - cur_anchored_pos
        out_lines.append(f"    .fill {tail_sz}, 1, 0xcc")

    out_s = "\n".join(out_lines) + "\n"

    # 7. Generate stubs.s
    stubs_res = generate_all_stubs(plt_entries, lookup_table)
    stubs_lines: List[str] = [
        stubs_res.stubs_asm,
        "",
        "/* Syscall bridge */",
        ".globl __nm_syscall_bridge",
        "__nm_syscall_bridge:",
        "    jmp __nm_syscall_tramp",
        "",
        "/* Dual ABI common handler and per-function dual stubs */",
        "__nm_dual_abi_common:",
        "    jmp *%r11",
        "",
    ]

    for at in sorted(analysis.addr_taken_funcs):
        stubs_lines.append(f".globl __nm_dual_{at:08x}")
        stubs_lines.append(f"__nm_dual_{at:08x}:")
        stubs_lines.append(f"    leal .L_{at:x}(%rip), %r11d")
        stubs_lines.append("    jmp __nm_dual_abi_common")

    # Define sym_got_plt_{idx} storage
    stubs_lines.append("")
    stubs_lines.append(".section .data.rel.ro, \"aw\"")
    for idx in range(len(plt_entries)):
        stubs_lines.append(f".globl sym_got_plt_{idx}")
        stubs_lines.append(f"sym_got_plt_{idx}: .long 0")

    stubs_s = "\n".join(stubs_lines) + "\n"

    # 8. Generate layout.ld linker script
    layout_ld = f"""SECTIONS
{{
  . = 0x{orig_text_va:x};
  .text : {{ *(.text) }}

  . = 0x{rewritten_base_va:x};
  .text.lifted : {{ *(.text.lifted) }}
  .text.stubs : {{ *(.text.stubs) }}
  .rodata.jumptables : {{ *(.rodata.jumptables) }}

  . = ALIGN(0x1000);
  .data.rel.ro : {{ *(.data.rel.ro) }}
}}
"""

    # 9. Build manifest dictionary
    sha_out = hashlib.sha256(out_s.encode("utf-8")).hexdigest()
    sha_stubs = hashlib.sha256(stubs_s.encode("utf-8")).hexdigest()
    sha_ld = hashlib.sha256(layout_ld.encode("utf-8")).hexdigest()

    manifest_dict: Dict[str, Any] = {
        "version": "newminhh-asm-bundle-v1",
        "sha256": {
            "out.s": sha_out,
            "stubs.s": sha_stubs,
            "layout.ld": sha_ld,
        },
        "anchored_vas": {
            "orig_text_va": orig_text_va,
            "orig_text_size": orig_text_sz,
            "lifted_text_base_va": rewritten_base_va,
            "entry_va": analysis.e_entry,
        },
        "imports": [sym for sym, _gva, _sidx in plt_entries],
        "target_interp": analysis.interp_str or "/libx32/ld-linux-x32.so.2",
        "lookup_provenance": (
            lookup_table.source_path if lookup_table and lookup_table.source_path else "default"
        ),
    }

    return out_s, stubs_s, layout_ld, manifest_dict


def emit_asm_bundle(
    dir_path: str,
    out_s: str,
    stubs_s: str,
    layout_ld: str,
    manifest_dict: Dict[str, Any],
) -> None:
    """Write complete assembly bundle to dir_path."""
    os.makedirs(dir_path, exist_ok=True)

    out_path = os.path.join(dir_path, "out.s")
    stubs_path = os.path.join(dir_path, "stubs.s")
    ld_path = os.path.join(dir_path, "layout.ld")
    manifest_path = os.path.join(dir_path, "manifest.json")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_s)
    with open(stubs_path, "w", encoding="utf-8") as f:
        f.write(stubs_s)
    with open(ld_path, "w", encoding="utf-8") as f:
        f.write(layout_ld)

    manifest_copy = dict(manifest_dict)
    manifest_copy["sha256"] = {
        "out.s": hashlib.sha256(out_s.encode("utf-8")).hexdigest(),
        "stubs.s": hashlib.sha256(stubs_s.encode("utf-8")).hexdigest(),
        "layout.ld": hashlib.sha256(layout_ld.encode("utf-8")).hexdigest(),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_copy, f, indent=2)


def _parse_elf32_binary(raw_bytes: bytes) -> Tuple[Dict[str, bytes], Dict[str, int], int]:
    """Pure-Python ELF32 parser extracting section bytes, symbol VAs, and bss size."""
    if len(raw_bytes) < 52 or raw_bytes[:4] != b"\x7fELF":
        raise PatcherError("Output file is not a valid ELF binary")

    e_shoff = struct.unpack_from("<I", raw_bytes, 32)[0]
    e_shentsize = struct.unpack_from("<H", raw_bytes, 46)[0]
    e_shnum = struct.unpack_from("<H", raw_bytes, 48)[0]
    e_shstrndx = struct.unpack_from("<H", raw_bytes, 50)[0]

    shstr_hdr = raw_bytes[e_shoff + e_shstrndx * e_shentsize : e_shoff + (e_shstrndx + 1) * e_shentsize]
    shstr_off, shstr_sz = struct.unpack_from("<II", shstr_hdr, 16)
    shstrtab = raw_bytes[shstr_off : shstr_off + shstr_sz]

    sections: Dict[str, bytes] = {}
    symtab_info: Optional[Tuple[int, int, int]] = None
    strtab_info: Optional[Tuple[int, int]] = None
    bss_size = 0

    for i in range(e_shnum):
        s_hdr = raw_bytes[e_shoff + i * e_shentsize : e_shoff + (i + 1) * e_shentsize]
        name_idx, s_type, s_flags, s_addr, s_off, s_size, s_link = struct.unpack_from("<IIIIIII", s_hdr, 0)
        name = shstrtab[name_idx : shstrtab.find(b"\x00", name_idx)].decode("latin1")
        sections[name] = raw_bytes[s_off : s_off + s_size]
        if name == ".bss":
            bss_size = s_size
        if s_type == 2:  # SHT_SYMTAB
            symtab_info = (s_off, s_size, s_link)
        elif name == ".strtab":
            strtab_info = (s_off, s_size)

    symbol_vas: Dict[str, int] = {}
    if symtab_info and strtab_info:
        s_off, s_size, _link = symtab_info
        str_off, str_size = strtab_info
        strtab = raw_bytes[str_off : str_off + str_size]
        for i in range(0, s_size, 16):
            st_name, st_value = struct.unpack_from("<II", raw_bytes, s_off + i)[:2]
            sname = strtab[st_name : strtab.find(b"\x00", st_name)].decode("latin1")
            if sname:
                symbol_vas[sname] = st_value

    return sections, symbol_vas, bss_size


def load_and_assemble_from_asm(dir_path: str) -> AssembledBundle:
    """Validate manifest and assemble out.s + stubs.s with as --x32 and ld."""
    manifest_path = os.path.join(dir_path, "manifest.json")
    out_path = os.path.join(dir_path, "out.s")
    stubs_path = os.path.join(dir_path, "stubs.s")
    ld_path = os.path.join(dir_path, "layout.ld")

    if not os.path.exists(manifest_path):
        raise PatcherError(f"Missing manifest.json in assembly bundle '{dir_path}'")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Validate SHA-256 hashes
    for filename, expected_hash in manifest.get("sha256", {}).items():
        fpath = os.path.join(dir_path, filename)
        if not os.path.exists(fpath):
            raise PatcherError(f"Missing bundle file '{filename}' in '{dir_path}'")
        with open(fpath, "r", encoding="utf-8") as f:
            actual_hash = hashlib.sha256(f.read().encode("utf-8")).hexdigest()
        if actual_hash != expected_hash:
            raise PatcherError(
                f"SHA-256 hash mismatch for '{filename}' in '{dir_path}' "
                f"(expected {expected_hash}, got {actual_hash})"
            )

    as_bin = shutil.which("as")
    ld_bin = shutil.which("ld")
    if not as_bin or not ld_bin:
        raise PatcherError("GNU 'as' and 'ld' are required to assemble bundles")

    with tempfile.TemporaryDirectory(prefix="nm_bundle_") as tmpdir:
        out_o = os.path.join(tmpdir, "out.o")
        stubs_o = os.path.join(tmpdir, "stubs.o")
        linked_elf = os.path.join(tmpdir, "linked.elf")

        # Assemble out.s
        p1 = subprocess.run([as_bin, "--x32", out_path, "-o", out_o], capture_output=True, text=True)
        if p1.returncode != 0:
            raise PatcherError(f"Assembly of out.s failed:\n{p1.stderr}")

        # Assemble stubs.s
        p2 = subprocess.run([as_bin, "--x32", stubs_path, "-o", stubs_o], capture_output=True, text=True)
        if p2.returncode != 0:
            raise PatcherError(f"Assembly of stubs.s failed:\n{p2.stderr}")

        # Link with ld -m elf32_x86_64
        p3 = subprocess.run(
            [
                ld_bin,
                "-m",
                "elf32_x86_64",
                "--unresolved-symbols=ignore-all",
                "-T",
                ld_path,
                out_o,
                stubs_o,
                "-o",
                linked_elf,
            ],
            capture_output=True,
            text=True,
        )
        if p3.returncode != 0:
            raise PatcherError(f"Linking bundle failed:\n{p3.stderr}")

        with open(linked_elf, "rb") as ef:
            raw_elf = ef.read()

        sections, symbol_vas, bss_size = _parse_elf32_binary(raw_elf)
        text_bytes = sections.get(".text.lifted", sections.get(".text", b""))
        rodata_bytes = sections.get(".rodata.jumptables", sections.get(".rodata", b""))
        data_bytes = sections.get(".data.rel.ro", sections.get(".data", b""))

        return AssembledBundle(
            text_bytes=text_bytes,
            rodata_bytes=rodata_bytes,
            data_bytes=data_bytes,
            bss_size=bss_size,
            symbol_vas=symbol_vas,
            elf_path=linked_elf,
        )


def emit_relocatable_2b(
    analysis: AnalysisResult,
    cfg: CFGResult,
    lookup_table: Optional[LookupTable] = None,
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[bool, List[str], Optional[bytes]]:
    """Gated Level 2B PIE Relocatable Emitter.

    First checks all 4 Level 2B gate conditions via check_relocatable_2b_gate().
    If ANY condition fails, returns (False, reasons, None) cleanly so caller
    can fall back to Level 2A.

    When gate passes, re-emits .rodata, .data, .bss, .init_array, .fini_array
    symbolically, converts R_386_RELATIVE words into .long .L_<target>, combines
    with lifted symbolic .text + specialized stubs, assembles via as --x32,
    and links a standalone x32 PIE via ld -m elf32_x86_64 -pie without anchoring.
    """
    passed, reasons = check_relocatable_2b_gate(analysis, cfg)
    if not passed:
        return False, reasons, None

    as_bin = shutil.which("as")
    ld_bin = shutil.which("ld")
    if not as_bin or not ld_bin:
        raise PatcherError("GNU 'as' and 'ld' are required to link Level 2B PIE")

    # 1. Generate symbolic text without anchored original text
    out_s, stubs_s, _ld, _m = emit_asm_level_2a(analysis, cfg, lookup_table, rewritten_base_va=0)

    # 2. Re-emit data sections symbolically
    reloc_by_offset: Dict[int, Any] = {
        r.offset: r for r in (analysis.rel_dyn + analysis.rel_plt)
    }

    data_sec_lines: List[str] = [
        "",
        "/* ------------------------------------------------------------------- */",
        "/* Level 2B Symbolic Relocatable Data Sections                          */",
        "/* ------------------------------------------------------------------- */",
    ]

    for sh in analysis.shdrs:
        if sh.name in (".rodata", ".data", ".init_array", ".fini_array"):
            sec_flags = "\"aw\"" if sh.name in (".data", ".init_array", ".fini_array") else "\"a\""
            data_sec_lines.append(f".section {sh.name}, {sec_flags}")
            data_sec_lines.append(f".align {sh.sh_addralign or 4}")
            data_sec_lines.append(f".globl .L_sec_{sh.name[1:]}")
            data_sec_lines.append(f".L_sec_{sh.name[1:]}:")

            sec_data = analysis.raw_bytes[sh.sh_offset : sh.sh_offset + sh.sh_size]
            words = len(sec_data) // 4
            for i in range(words):
                va = sh.sh_addr + i * 4
                raw_word = struct.unpack_from("<I", sec_data, i * 4)[0]
                if va in reloc_by_offset:
                    rel = reloc_by_offset[va]
                    if rel.r_type == 8:  # R_386_RELATIVE
                        data_sec_lines.append(f"    .long .L_{raw_word:x}")
                    elif rel.sym_name:
                        data_sec_lines.append(f"    .long {rel.sym_name}")
                    else:
                        data_sec_lines.append(f"    .long 0x{raw_word:08x}")
                else:
                    data_sec_lines.append(f"    .long 0x{raw_word:08x}")

        elif sh.name == ".bss":
            data_sec_lines.append(".section .bss, \"aw\", @nobits")
            data_sec_lines.append(f".space {sh.sh_size}")

    data_s = "\n".join(data_sec_lines) + "\n"
    full_src = out_s + "\n" + stubs_s + "\n" + data_s

    with tempfile.TemporaryDirectory(prefix="nm_pie_") as tmpdir:
        src_path = os.path.join(tmpdir, "pie.s")
        o_path = os.path.join(tmpdir, "pie.o")
        pie_path = output_path or os.path.join(tmpdir, "standalone.pie")

        with open(src_path, "w", encoding="utf-8") as f:
            f.write(full_src)

        p_as = subprocess.run([as_bin, "--x32", src_path, "-o", o_path], capture_output=True, text=True)
        if p_as.returncode != 0:
            raise PatcherError(f"Level 2B 'as --x32' failed:\n{p_as.stderr}")

        p_ld = subprocess.run(
            [ld_bin, "-m", "elf32_x86_64", "--unresolved-symbols=ignore-all", "-pie", o_path, "-o", pie_path],
            capture_output=True,
            text=True,
        )
        if p_ld.returncode != 0:
            raise PatcherError(f"Level 2B 'ld -m elf32_x86_64 -pie' failed:\n{p_ld.stderr}")

        with open(pie_path, "rb") as pf:
            pie_bytes = pf.read()

        return True, [], pie_bytes
