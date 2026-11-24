"""Control Flow Graph (CFG) extraction and Level 2B PIE gate verification.

Builds recursive-descent CFG with linear-sweep gap fill over i386 executable
code regions from AnalysisResult, recording undecodable gaps as DataGap
and validating 4 Level 2B gate conditions for relocatable PIE generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import struct
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
    from .analyzer import AnalysisResult, CodeRange, JumpTableInfo, DynSymInfo, RelocInfo, ShdrInfo
except ImportError:
    pass

# Standard noreturn symbols matching analyzer.py
NORETURN_SYMS = {
    "__libc_start_main",
    "exit",
    "_exit",
    "abort",
    "__stack_chk_fail",
    "__stack_chk_fail_local",
    "pthread_exit",
}

# Standard NOP / INT3 byte values and known multi-byte NOP prefixes
PADDING_BYTES = {0x90, 0xCC}
MULTIBYTE_NOP_PATTERNS = (
    b"\x8d\x76\x00",              # lea 0(%esi), %esi (3B)
    b"\x8d\x74\x26\x00",          # lea 0(%esi,%eiz,1), %esi (4B)
    b"\x90\x8d\x74\x26\x00",      # nop; lea (5B)
    b"\x8d\xb6\x00\x00\x00\x00",  # lea 0(%esi), %esi (6B)
    b"\x8d\xb4\x26\x00\x00\x00\x00",  # 7B
    b"\x0f\x1f\x00",              # nopl (%eax)
    b"\x0f\x1f\x40\x00",          # nopl 0(%eax)
    b"\x0f\x1f\x44\x00\x00",      # nopl 0(%eax,%eax,1)
    b"\x0f\x1f\x80\x00\x00\x00\x00",
    b"\x0f\x1f\x84\x00\x00\x00\x00\x00",
)


@dataclass
class BasicBlock:
    """A straight-line sequence of instructions with single entry and single exit."""

    start_va: int
    end_va: int
    insns: List[Any]  # List[capstone.CsInsn]
    successors: List[int] = field(default_factory=list)
    is_func_entry: bool = False


@dataclass
class DataGap:
    """An unreachable, unaligned, or undecodable data region between basic blocks."""

    start_va: int
    end_va: int
    raw_bytes: bytes


@dataclass
class CFGResult:
    """Complete CFG analysis result over all executable code ranges."""

    blocks: List[BasicBlock]
    gaps: List[DataGap]
    block_map: Dict[int, BasicBlock]  # start_va -> BasicBlock
    func_entries: Set[int]
    coverage_ratio: float
    unresolved_indirect_jumps: List[int]


def _is_all_padding(data: bytes) -> bool:
    """Check if byte sequence consists solely of NOP/INT3 or multi-byte NOPs."""
    if not data:
        return True
    idx = 0
    length = len(data)
    while idx < length:
        b = data[idx]
        if b in PADDING_BYTES:
            idx += 1
            continue
        # Check multibyte NOPs
        matched = False
        for pat in MULTIBYTE_NOP_PATTERNS:
            if data.startswith(pat, idx):
                idx += len(pat)
                matched = True
                break
        if matched:
            continue
        return False
    return True


def build_cfg(analysis: AnalysisResult) -> CFGResult:
    """Construct CFG over i386 executable code regions from AnalysisResult.

    Uses recursive descent from seeds followed by linear-sweep gap fill.
    Never silently truncates or misdecodes inline data.
    """
    if capstone is None:
        raise PatcherError(
            "capstone library is required to build CFG (available on host or in re-env)"
        )

    # 1. Collect initial seed entry points
    seeds: Set[int] = set()

    def _in_code_ranges(va: int) -> bool:
        return any(cr.va <= va < cr.va + len(cr.data) for cr in analysis.code_ranges)

    if analysis.e_entry and _in_code_ranges(analysis.e_entry):
        seeds.add(analysis.e_entry)

    # Exported functions and symbols
    for sym in analysis.dynsyms:
        if sym.st_value != 0 and (sym.sym_type in (1, 2, 10) or sym.st_shndx != 0):
            if _in_code_ranges(sym.st_value):
                seeds.add(sym.st_value)

    if analysis.init_va and _in_code_ranges(analysis.init_va):
        seeds.add(analysis.init_va)
    if analysis.fini_va and _in_code_ranges(analysis.fini_va):
        seeds.add(analysis.fini_va)

    for a in analysis.init_array:
        if a and _in_code_ranges(a):
            seeds.add(a)
    for a in analysis.fini_array:
        if a and _in_code_ranges(a):
            seeds.add(a)

    for at in analysis.addr_taken_funcs:
        if at and _in_code_ranges(at):
            seeds.add(at)

    for jt in analysis.jump_tables.values():
        for tgt in jt.targets:
            if tgt and _in_code_ranges(tgt):
                seeds.add(tgt)

    for fn in analysis.functions:
        if fn and _in_code_ranges(fn):
            seeds.add(fn)

    func_entries: Set[int] = set(seeds)

    # 2. Recursive descent disassembly using capstone
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
    md.detail = True
    try:
        md.syntax = capstone.CS_OPT_SYNTAX_ATT
    except Exception:
        pass

    def _is_noreturn(insn: Any) -> bool:
        if insn.id != cs_x86.X86_INS_CALL or len(insn.operands) != 1:
            return False
        op = insn.operands[0]
        if op.type == cs_x86.X86_OP_IMM:
            tgt = op.imm & 0xFFFFFFFF
            stub = analysis.plt_stubs.get(tgt)
            if stub and stub.sym_name in NORETURN_SYMS:
                return True
            sname = analysis.sym_names.get(tgt, "")
            if sname in NORETURN_SYMS:
                return True
        return False

    def _get_cr(va: int) -> Optional[CodeRange]:
        return next((c for c in analysis.code_ranges if c.va <= va < c.va + len(c.data)), None)

    worklist: List[int] = sorted(seeds)
    visited_insns: Dict[int, Any] = {}
    insn_successors: Dict[int, List[int]] = {}
    unresolved_indirect_jumps: List[int] = []
    branch_targets: Set[int] = set(seeds)

    while worklist:
        cur_va = worklist.pop()
        if cur_va in visited_insns or not _in_code_ranges(cur_va):
            continue

        cr = _get_cr(cur_va)
        if cr is None:
            continue
        off = cur_va - cr.va
        insn_stream = md.disasm(cr.data[off:], cur_va)

        for insn in insn_stream:
            va = insn.address
            if va in visited_insns:
                # Met previously disassembled code
                break

            visited_insns[va] = insn
            succs: List[int] = []

            # 2a. RET / RETF / IRET -> terminator, no successors
            if insn.id in (cs_x86.X86_INS_RET, cs_x86.X86_INS_RETF, cs_x86.X86_INS_IRET, cs_x86.X86_INS_IRETD):
                insn_successors[va] = []
                break

            # 2b. JMP unconditional direct
            if insn.id == cs_x86.X86_INS_JMP:
                if len(insn.operands) == 1 and insn.operands[0].type == cs_x86.X86_OP_IMM:
                    tgt = insn.operands[0].imm & 0xFFFFFFFF
                    succs.append(tgt)
                    branch_targets.add(tgt)
                    if tgt not in visited_insns and _in_code_ranges(tgt):
                        worklist.append(tgt)
                elif va in analysis.jump_tables:
                    jt = analysis.jump_tables[va]
                    for tgt in jt.targets:
                        succs.append(tgt)
                        branch_targets.add(tgt)
                        if tgt not in visited_insns and _in_code_ranges(tgt):
                            worklist.append(tgt)
                elif va in analysis.plt_stubs:
                    # PLT tail jump to external library
                    succs = []
                else:
                    # Check if PLT tail jump via GOT
                    op0 = insn.operands[0] if insn.operands else None
                    is_plt_jmp = False
                    if op0 is not None and op0.type == cs_x86.X86_OP_MEM:
                        slot_va = None
                        if op0.mem.base == cs_x86.X86_REG_INVALID:
                            slot_va = op0.mem.disp & 0xFFFFFFFF
                        elif analysis.got_va != 0 and op0.mem.base == cs_x86.X86_REG_EBX:
                            slot_va = (analysis.got_va + op0.mem.disp) & 0xFFFFFFFF
                        if slot_va is not None:
                            rel = next(
                                (r for r in analysis.rel_plt + analysis.rel_dyn if r.offset == slot_va),
                                None,
                            )
                            if rel and rel.sym_name:
                                is_plt_jmp = True
                    if not is_plt_jmp:
                        unresolved_indirect_jumps.append(va)
                insn_successors[va] = succs
                break

            # 2c. JCC conditional branches
            if insn.id in (
                cs_x86.X86_INS_JA, cs_x86.X86_INS_JAE, cs_x86.X86_INS_JB, cs_x86.X86_INS_JBE,
                cs_x86.X86_INS_JCXZ, cs_x86.X86_INS_JECXZ, cs_x86.X86_INS_JE, cs_x86.X86_INS_JG,
                cs_x86.X86_INS_JGE, cs_x86.X86_INS_JL, cs_x86.X86_INS_JLE, cs_x86.X86_INS_JNE,
                cs_x86.X86_INS_JNO, cs_x86.X86_INS_JNP, cs_x86.X86_INS_JNS, cs_x86.X86_INS_JO,
                cs_x86.X86_INS_JP, cs_x86.X86_INS_JS, cs_x86.X86_INS_LOOP, cs_x86.X86_INS_LOOPE,
                cs_x86.X86_INS_LOOPNE,
            ):
                next_va = va + insn.size
                succs.append(next_va)
                branch_targets.add(next_va)
                if len(insn.operands) == 1 and insn.operands[0].type == cs_x86.X86_OP_IMM:
                    tgt = insn.operands[0].imm & 0xFFFFFFFF
                    succs.append(tgt)
                    branch_targets.add(tgt)
                    if tgt not in visited_insns and _in_code_ranges(tgt):
                        worklist.append(tgt)
                insn_successors[va] = succs
                # Split after conditional branch
                break

            # 2d. CALL
            if insn.id == cs_x86.X86_INS_CALL:
                next_va = va + insn.size
                if len(insn.operands) == 1 and insn.operands[0].type == cs_x86.X86_OP_IMM:
                    tgt = insn.operands[0].imm & 0xFFFFFFFF
                    func_entries.add(tgt)
                    branch_targets.add(tgt)
                    if tgt not in visited_insns and _in_code_ranges(tgt):
                        worklist.append(tgt)
                if _is_noreturn(insn):
                    insn_successors[va] = []
                    break
                else:
                    succs.append(next_va)
                    branch_targets.add(next_va)
                    insn_successors[va] = succs
                    # Continue linearly along caller
                    continue

            # 2e. Regular instruction
            next_va = va + insn.size
            succs.append(next_va)
            insn_successors[va] = succs

    # 3. Form BasicBlock objects by splitting at leaders
    leaders = set(branch_targets) | set(func_entries)
    sorted_vas = sorted(visited_insns.keys())

    blocks: List[BasicBlock] = []
    block_map: Dict[int, BasicBlock] = {}
    curr_block_insns: List[Any] = []

    def _flush_block() -> None:
        if not curr_block_insns:
            return
        b_start = curr_block_insns[0].address
        b_end = curr_block_insns[-1].address + curr_block_insns[-1].size
        last_insn = curr_block_insns[-1]
        succs = list(insn_successors.get(last_insn.address, []))
        bb = BasicBlock(
            start_va=b_start,
            end_va=b_end,
            insns=list(curr_block_insns),
            successors=succs,
            is_func_entry=(b_start in func_entries),
        )
        blocks.append(bb)
        block_map[b_start] = bb
        curr_block_insns.clear()

    for idx, va in enumerate(sorted_vas):
        if va in leaders and curr_block_insns:
            _flush_block()
        insn = visited_insns[va]
        curr_block_insns.append(insn)

        # Check if this instruction ends a block
        is_terminator = (
            insn.id in (cs_x86.X86_INS_RET, cs_x86.X86_INS_RETF, cs_x86.X86_INS_IRET, cs_x86.X86_INS_IRETD, cs_x86.X86_INS_JMP)
            or (insn.id in (
                cs_x86.X86_INS_JA, cs_x86.X86_INS_JAE, cs_x86.X86_INS_JB, cs_x86.X86_INS_JBE,
                cs_x86.X86_INS_JCXZ, cs_x86.X86_INS_JECXZ, cs_x86.X86_INS_JE, cs_x86.X86_INS_JG,
                cs_x86.X86_INS_JGE, cs_x86.X86_INS_JL, cs_x86.X86_INS_JLE, cs_x86.X86_INS_JNE,
                cs_x86.X86_INS_JNO, cs_x86.X86_INS_JNP, cs_x86.X86_INS_JNS, cs_x86.X86_INS_JO,
                cs_x86.X86_INS_JP, cs_x86.X86_INS_JS, cs_x86.X86_INS_LOOP, cs_x86.X86_INS_LOOPE,
                cs_x86.X86_INS_LOOPNE,
            ))
            or _is_noreturn(insn)
        )
        if is_terminator:
            _flush_block()
        elif idx + 1 < len(sorted_vas) and sorted_vas[idx + 1] in leaders:
            _flush_block()

    _flush_block()

    # 4. Linear-sweep gap fill over executable sections
    blocks.sort(key=lambda b: b.start_va)
    gaps: List[DataGap] = []
    extra_blocks: List[BasicBlock] = []

    for cr in analysis.code_ranges:
        cr_start = cr.va
        cr_end = cr.va + len(cr.data)
        cr_blocks = [b for b in blocks if cr_start <= b.start_va < cr_end]

        curr_pos = cr_start
        for b in cr_blocks:
            if b.start_va > curr_pos:
                gap_start = curr_pos
                gap_end = b.start_va
                gap_bytes = cr.data[gap_start - cr_start : gap_end - cr_start]

                # Attempt linear sweep disassembly across gap
                if not _is_all_padding(gap_bytes):
                    gap_insns = list(md.disasm(gap_bytes, gap_start))
                    if gap_insns:
                        # Split valid decoded instructions into basic block
                        sub_start = gap_insns[0].address
                        sub_end = gap_insns[-1].address + gap_insns[-1].size
                        extra_bb = BasicBlock(
                            start_va=sub_start,
                            end_va=sub_end,
                            insns=gap_insns,
                            successors=[sub_end],
                            is_func_entry=(sub_start in func_entries),
                        )
                        extra_blocks.append(extra_bb)
                        # Record any remainder of gap as DataGap
                        if sub_end < gap_end:
                            tail_bytes = cr.data[sub_end - cr_start : gap_end - cr_start]
                            gaps.append(DataGap(start_va=sub_end, end_va=gap_end, raw_bytes=tail_bytes))
                    else:
                        gaps.append(DataGap(start_va=gap_start, end_va=gap_end, raw_bytes=gap_bytes))
                else:
                    gaps.append(DataGap(start_va=gap_start, end_va=gap_end, raw_bytes=gap_bytes))

            curr_pos = max(curr_pos, b.end_va)

        if curr_pos < cr_end:
            gap_start = curr_pos
            gap_end = cr_end
            gap_bytes = cr.data[gap_start - cr_start : gap_end - cr_start]
            if not _is_all_padding(gap_bytes):
                gap_insns = list(md.disasm(gap_bytes, gap_start))
                if gap_insns:
                    sub_start = gap_insns[0].address
                    sub_end = gap_insns[-1].address + gap_insns[-1].size
                    extra_bb = BasicBlock(
                        start_va=sub_start,
                        end_va=sub_end,
                        insns=gap_insns,
                        successors=[],
                        is_func_entry=(sub_start in func_entries),
                    )
                    extra_blocks.append(extra_bb)
                    if sub_end < gap_end:
                        tail_bytes = cr.data[sub_end - cr_start : gap_end - cr_start]
                        gaps.append(DataGap(start_va=sub_end, end_va=gap_end, raw_bytes=tail_bytes))
                else:
                    gaps.append(DataGap(start_va=gap_start, end_va=gap_end, raw_bytes=gap_bytes))
            else:
                gaps.append(DataGap(start_va=gap_start, end_va=gap_end, raw_bytes=gap_bytes))

    # Merge extra blocks and sort
    blocks.extend(extra_blocks)
    blocks.sort(key=lambda b: b.start_va)
    for b in extra_blocks:
        block_map[b.start_va] = b

    # 5. Compute coverage ratio
    total_exec_bytes = sum(len(cr.data) for cr in analysis.code_ranges)
    covered_bytes = sum(b.end_va - b.start_va for b in blocks)
    coverage_ratio = (covered_bytes / total_exec_bytes) if total_exec_bytes > 0 else 1.0

    return CFGResult(
        blocks=blocks,
        gaps=gaps,
        block_map=block_map,
        func_entries=func_entries,
        coverage_ratio=coverage_ratio,
        unresolved_indirect_jumps=unresolved_indirect_jumps,
    )


def check_relocatable_2b_gate(
    analysis: AnalysisResult, cfg: CFGResult
) -> Tuple[bool, List[str]]:
    """Check all 4 Level 2B gate conditions for relocatable PIE generation.

    Conditions:
      1. analysis.e_type == ET_DYN (3) and NOT DT_TEXTREL / DF_TEXTREL
      2. Every absolute data pointer in .data / .rodata / .init_array / .fini_array
         is backed by a relocation (R_386_RELATIVE / R_386_32 / R_386_GLOB_DAT).
      3. Every indirect jump / jump table is resolved as GOT-relative
         (zero unresolved indirect jumps).
      4. CFG executable byte coverage == 100% modulo nop/int3 alignment padding.

    Returns:
      (True, []) if all 4 conditions pass, or (False, reasons) if any condition fails.
    """
    reasons: List[str] = []

    # -------------------------------------------------------------------------
    # Gate Condition 1: ET_DYN and no DT_TEXTREL
    # -------------------------------------------------------------------------
    if analysis.e_type != 3:  # ET_DYN == 3
        reasons.append(f"ELF e_type is {analysis.e_type} (expected ET_DYN == 3)")

    # DT_TEXTREL is tag 22; DF_TEXTREL is 0x4 in DT_FLAGS (tag 30)
    has_textrel = (22 in analysis.dyn_tags) or bool(analysis.dyn_tags.get(30, 0) & 0x4)
    if has_textrel:
        reasons.append("ELF contains DT_TEXTREL / DF_TEXTREL (text relocations present)")

    # -------------------------------------------------------------------------
    # Gate Condition 2: Absolute data pointers backed by relocations
    # -------------------------------------------------------------------------
    # Relocation types: R_386_32 (1), R_386_GLOB_DAT (6), R_386_RELATIVE (8)
    reloc_offsets: Set[int] = {
        r.offset for r in (analysis.rel_dyn + analysis.rel_plt) if r.r_type in (1, 6, 8)
    }

    # Minimum and maximum mapped VAs for detecting pointer values
    min_va = min((p.p_vaddr for p in analysis.phdrs if p.p_memsz > 0), default=0)
    max_va = analysis.max_vaddr

    checked_sections = {".data", ".rodata", ".init_array", ".fini_array"}
    for sh in analysis.shdrs:
        if sh.name not in checked_sections or sh.sh_size < 4:
            continue
        # Read 4-byte words
        sec_data = analysis.raw_bytes[sh.sh_offset : sh.sh_offset + sh.sh_size]
        num_words = len(sec_data) // 4
        for i in range(num_words):
            word_va = sh.sh_addr + i * 4
            word_val = struct.unpack_from("<I", sec_data, i * 4)[0]

            # In .init_array / .fini_array, every non-zero entry MUST be relocated
            if sh.name in (".init_array", ".fini_array"):
                if word_val != 0 and word_val != 0xFFFFFFFF and word_va not in reloc_offsets:
                    reasons.append(
                        f"Unrelocated function pointer 0x{word_val:08x} at 0x{word_va:08x} in {sh.name}"
                    )
            else:
                # In .data / .rodata, if the word points inside the binary's address space
                # but has no relocation, it's an un-relocatable absolute address!
                if (
                    min_va <= word_val < max_va
                    and word_val != 0
                    and word_val != 0xFFFFFFFF
                    and word_va not in reloc_offsets
                ):
                    reasons.append(
                        f"Unrelocated absolute data pointer 0x{word_val:08x} at 0x{word_va:08x} in {sh.name}"
                    )

    # -------------------------------------------------------------------------
    # Gate Condition 3: Indirect jumps / jump tables resolved as GOT-relative
    # -------------------------------------------------------------------------
    if cfg.unresolved_indirect_jumps:
        hex_vas = [f"0x{v:08x}" for v in cfg.unresolved_indirect_jumps]
        reasons.append(f"Unresolved indirect jumps present: {hex_vas}")

    for jt in analysis.jump_tables.values():
        if not jt.is_pic or jt.got_reg is None:
            reasons.append(
                f"Jump table at 0x{jt.jmp_va:08x} is not GOT-relative (is_pic={jt.is_pic}, got_reg={jt.got_reg})"
            )

    # -------------------------------------------------------------------------
    # Gate Condition 4: Executable byte coverage == 100% modulo padding
    # -------------------------------------------------------------------------
    non_padding_gaps: List[DataGap] = []
    for gap in cfg.gaps:
        if not _is_all_padding(gap.raw_bytes):
            non_padding_gaps.append(gap)

    if non_padding_gaps:
        gap_desc = [
            f"0x{g.start_va:08x}-0x{g.end_va:08x} ({len(g.raw_bytes)}B: {g.raw_bytes[:8].hex()}...)"
            for g in non_padding_gaps[:5]
        ]
        reasons.append(
            f"CFG executable coverage has non-padding undecoded gaps: {len(non_padding_gaps)} gap(s): {gap_desc}"
        )

    return (len(reasons) == 0, reasons)
