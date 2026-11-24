"""Stage 2b — Out-of-place i386 -> Long Mode (x32) 4-byte stack-stride rewriter."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Dict, List, Optional, Tuple

import capstone
from capstone import x86 as cs_x86

from . import PatcherError
from .analyzer import AnalysisResult, GotProvenanceTracker, JumpTableInfo

REG32_IDX: Dict[int, int] = {
    cs_x86.X86_REG_EAX: 0,
    cs_x86.X86_REG_ECX: 1,
    cs_x86.X86_REG_EDX: 2,
    cs_x86.X86_REG_EBX: 3,
    cs_x86.X86_REG_ESP: 4,
    cs_x86.X86_REG_EBP: 5,
    cs_x86.X86_REG_ESI: 6,
    cs_x86.X86_REG_EDI: 7,
}

HI8_XCHG: Dict[int, bytes] = {
    cs_x86.X86_REG_AH: b"\x86\xc4",  # xchg %al, %ah
    cs_x86.X86_REG_CH: b"\x86\xcd",  # xchg %cl, %ch
    cs_x86.X86_REG_DH: b"\x86\xd6",  # xchg %dl, %dh
    cs_x86.X86_REG_BH: b"\x86\xdf",  # xchg %bl, %bh
}

REG32_NAME_TO_IDX: Dict[str, int] = {
    "eax": 0,
    "ecx": 1,
    "edx": 2,
    "ebx": 3,
    "esp": 4,
    "ebp": 5,
    "esi": 6,
    "edi": 7,
}

JCC_SHORT_MAP = {
    cs_x86.X86_INS_JO: 0x0,
    cs_x86.X86_INS_JNO: 0x1,
    cs_x86.X86_INS_JB: 0x2,
    cs_x86.X86_INS_JAE: 0x3,
    cs_x86.X86_INS_JE: 0x4,
    cs_x86.X86_INS_JNE: 0x5,
    cs_x86.X86_INS_JBE: 0x6,
    cs_x86.X86_INS_JA: 0x7,
    cs_x86.X86_INS_JS: 0x8,
    cs_x86.X86_INS_JNS: 0x9,
    cs_x86.X86_INS_JP: 0xA,
    cs_x86.X86_INS_JNP: 0xB,
    cs_x86.X86_INS_JL: 0xC,
    cs_x86.X86_INS_JGE: 0xD,
    cs_x86.X86_INS_JLE: 0xE,
    cs_x86.X86_INS_JG: 0xF,
}


@dataclass
class BranchFixup:
    patch_off: int       # offset inside rewritten_code of 4B rel32
    next_insn_va: int    # VA after the jmp/jcc/lea(%rip)
    target_orig_va: int  # original i386 VA or negative sentinel
    target_kind: str     # "orig" | "dispatch" | "label"
    target_label: str = ""


@dataclass
class JumpTableBlob:
    rodata_offset: int
    targets: List[int]   # original target VAs


@dataclass
class RewrittenOutput:
    code_bytes: bytearray
    rewritten_base_va: int
    orig_to_new: Dict[int, int]
    jump_table_blobs: List[JumpTableBlob]
    rodata_fixups: List[Tuple[int, int, int]]  # (patch_off_in_code, next_insn_va, rodata_off)
    label_fixups: List[Tuple[int, int, str]]   # (patch_off_in_code, next_insn_va, label_name)
    plt_entries: List[Tuple[str, int, int]]    # (orig_sym_name, got_slot_va, dispatch_slot_idx)
    fallback_funcs: List[Tuple[str, int]]      # (label_name, orig_va)
    rodata_jump_tables_size: int


def _emit_push_reg32(r: int) -> bytes:
    if r == 4:  # esp
        return b"\x41\x89\xe3\x48\x8d\x64\x24\xfc\x44\x89\x1c\x24"
    return b"\x48\x8d\x64\x24\xfc" + bytes([0x89, 0x04 | (r << 3), 0x24])


def _emit_pop_reg32(r: int) -> bytes:
    if r == 4:  # esp
        return b"\x8b\x24\x24"
    return bytes([0x8b, 0x04 | (r << 3), 0x24]) + b"\x48\x8d\x64\x24\x04"


def _emit_push_imm32(imm: int) -> bytes:
    return b"\x48\x8d\x64\x24\xfc\xc7\x04\x24" + struct.pack("<i", imm)


def _emit_lea_rip_to_reg32(r: int, disp32: int = 0) -> bytes:
    # leal disp32(%rip), %reg32 -> 8d <00|r<<3|5> disp32 (6 bytes)
    return bytes([0x8D, (r << 3) | 5]) + struct.pack("<i", disp32)


def _emit_lea_rip_to_r11d(disp32: int = 0) -> bytes:
    # leal disp32(%rip), %r11d -> 44 8d 1d disp32 (7 bytes)
    return b"\x44\x8d\x1d" + struct.pack("<i", disp32)


def _emit_lea_rip_to_r10d(disp32: int = 0) -> bytes:
    # leal disp32(%rip), %r10d -> 44 8d 15 disp32 (7 bytes)
    return b"\x44\x8d\x15" + struct.pack("<i", disp32)


def _fix_modrm_disp32_and_addr32(
    raw: bytes, modrm_off: int, is_gs: bool, add_addr32: bool
) -> bytes:
    """Fix ModRM mod=00 rm=5 (disp32 -> SIB 0x25) and prepend 0x67 addr32 if non-GS."""
    buf = bytearray(raw)
    if modrm_off > 0 and modrm_off < len(buf):
        modrm = buf[modrm_off]
        mod = (modrm >> 6) & 3
        reg = (modrm >> 3) & 7
        rm = modrm & 7
        if mod == 0 and rm == 5:
            buf[modrm_off] = (reg << 3) | 4
            buf.insert(modrm_off + 1, 0x25)
    if add_addr32 and not is_gs and (not buf or buf[0] != 0x67):
        return b"\x67" + bytes(buf)
    return bytes(buf)


def _rewrite_rm32_mov_r11d(
    insn: capstone.CsInsn, store_to_mem: bool, has_imm32: bool = False
) -> bytes:
    """Emit `movl <r/m32>, %r11d` (store_to_mem=False) or `movl %r11d, <r/m32>` (True)."""
    raw = bytearray(insn.bytes)
    modrm_off = insn.encoding.modrm_offset
    if modrm_off <= 0 or modrm_off >= len(raw):
        raise PatcherError(f"cannot extract ModRM for r/m32 at 0x{insn.address:08x}")
    # Preserve segment/66 prefixes if present
    prefixes = bytearray()
    for p in insn.prefix:
        if p and p not in (0x67,):
            prefixes.append(p)
    is_gs = 0x65 in prefixes
    modrm = raw[modrm_off]
    mod = (modrm >> 6) & 3
    rm = modrm & 7
    tail = bytearray(raw[modrm_off:-4] if has_imm32 else raw[modrm_off:])
    tail[0] = (modrm & 0xC7) | (3 << 3)  # reg = 3 -> %r11d with REX.R
    if mod == 0 and rm == 5:
        tail[0] = (0 << 6) | (3 << 3) | 4
        tail.insert(1, 0x25)
    out = bytearray()
    if mod != 3 and not is_gs:
        out.append(0x67)
    out.extend(prefixes)
    out.append(0x44)  # REX.R
    out.append(0x89 if store_to_mem else 0x8B)
    out.extend(tail)
    return bytes(out)


def _rewrite_moffs(insn: capstone.CsInsn) -> bytes:
    """Convert i386 A0..A3 moffs32 into ModRM+SIB 0x25 form safe for Long Mode."""
    raw = insn.bytes
    prefixes = bytearray()
    i = 0
    while i < len(raw) and raw[i] in (0x66, 0x67, 0x64, 0x65, 0x26, 0x2E, 0x36, 0x3E, 0xF0, 0xF2, 0xF3):
        if raw[i] != 0x67:
            prefixes.append(raw[i])
        i += 1
    op = raw[i]
    disp32 = raw[i + 1 : i + 5]
    if len(disp32) < 4:
        disp32 = disp32.ljust(4, b"\x00")
    new_op = {0xA0: 0x8A, 0xA1: 0x8B, 0xA2: 0x88, 0xA3: 0x89}[op]
    is_gs = 0x65 in prefixes
    out = bytearray()
    if not is_gs:
        out.append(0x67)
    out.extend(prefixes)
    out.extend([new_op, 0x04, 0x25])
    out.extend(disp32)
    return bytes(out)


def _rewrite_gs_insn(insn: capstone.CsInsn) -> bytes:
    """Safely rewrite %gs:-prefixed instructions for x32 Long Mode."""
    raw = bytes(insn.bytes)
    # Check A0..A3 moffs
    for idx, b in enumerate(raw):
        if b in (0x65, 0x66, 0x67, 0xF0, 0xF2, 0xF3):
            continue
        if b in (0xA0, 0xA1, 0xA2, 0xA3):
            return _rewrite_moffs(insn)
        break

    modrm_off = insn.encoding.modrm_offset
    if modrm_off > 0 and modrm_off < len(raw):
        modrm = raw[modrm_off]
        mod = (modrm >> 6) & 3
        rm = modrm & 7
        if mod == 0 and rm == 5:
            # %gs:disp32 -> convert rm=5 to SIB 0x25 (no 0x67 so disp32 sign-extends!)
            return _fix_modrm_disp32_and_addr32(raw, modrm_off, is_gs=True, add_addr32=False)

    # %gs:(%reg) or %gs:disp(%reg) -> sign-extend base reg into %r11
    mem_op = next((op for op in insn.operands if op.type == cs_x86.X86_OP_MEM), None)
    if (
        mem_op is not None
        and mem_op.mem.base != cs_x86.X86_REG_INVALID
        and mem_op.mem.index == cs_x86.X86_REG_INVALID
        and modrm_off > 0
    ):
        b_idx = REG32_IDX.get(mem_op.mem.base)
        if b_idx is not None:
            # Emit: movslq %base32, %r11 (4c 63 <d8+b_idx>)
            pre = bytes([0x4C, 0x63, 0xD8 + b_idx])
            buf = bytearray(raw)
            modrm = buf[modrm_off]
            mod = (modrm >> 6) & 3
            reg = (modrm >> 3) & 7
            orig_rm = modrm & 7

            # Check if instruction has a hi8 register operand (ah, ch, dh, bh)
            hi8_reg = next(
                (op.reg for op in insn.operands if op.type == cs_x86.X86_OP_REG and op.reg in HI8_XCHG),
                None,
            )

            # If original had SIB (orig_rm == 4, e.g. (%esp)), drop the SIB byte
            if orig_rm == 4 and modrm_off + 1 < len(buf):
                del buf[modrm_off + 1]

            if hi8_reg is not None:
                # hi8 reg (ah=4, ch=5, dh=6, bh=7) conflicts with REX prefix in Long Mode.
                # Remap ModRM reg to low8 register (al=0, cl=1, dl=2, bl=3) and wrap
                # with xchg %r_low, %hi8 (which preserves EFLAGS and all other registers).
                xchg_seq = HI8_XCHG[hi8_reg]
                new_reg = reg & 3
                buf[modrm_off] = (mod << 6) | (new_reg << 3) | 3  # rm = 3 -> %r11 with REX.B
                op_off = 0
                while op_off < modrm_off and buf[op_off] in (0x65, 0x66, 0x67, 0xF0, 0xF2, 0xF3):
                    op_off += 1
                buf.insert(op_off, 0x41)  # REX.B
                return pre + xchg_seq + bytes(buf) + xchg_seq

            buf[modrm_off] = (mod << 6) | (reg << 3) | 3  # rm = 3 -> %r11 with REX.B
            # Insert REX.B (0x41) right before opcode (after prefix bytes)
            op_off = 0
            while op_off < modrm_off and buf[op_off] in (0x65, 0x66, 0x67, 0xF0, 0xF2, 0xF3):
                op_off += 1
            buf.insert(op_off, 0x41)
            return pre + bytes(buf)

    return _fix_modrm_disp32_and_addr32(raw, modrm_off, is_gs=True, add_addr32=False)


def _is_gcc_multibyte_nop(insn: capstone.CsInsn) -> bool:
    if insn.id in (cs_x86.X86_INS_NOP, cs_x86.X86_INS_FNOP):
        return True
    if insn.id == cs_x86.X86_INS_XCHG and len(insn.operands) == 2:
        o0, o1 = insn.operands
        if o0.type == cs_x86.X86_OP_REG and o1.type == cs_x86.X86_OP_REG and o0.reg == o1.reg:
            return True
    if insn.id == cs_x86.X86_INS_LEA and len(insn.operands) == 2:
        o0, o1 = insn.operands
        if (
            o0.type == cs_x86.X86_OP_REG
            and o1.type == cs_x86.X86_OP_MEM
            and o1.mem.base == o0.reg
            and o1.mem.disp == 0
            and (o1.mem.index == cs_x86.X86_REG_INVALID or o1.mem.index == cs_x86.X86_REG_EIZ)
        ):
            return True
    return False


def rewrite_binary(
    analysis: AnalysisResult,
    hybrid_mode: str = "auto",
    verbose: bool = False,
    rewritten_base_va: int = 0,
) -> RewrittenOutput:
    """Re-encode all functions in analysis.code_ranges into Long Mode x32 bytes.

    If rewritten_base_va is passed as 0, Pass 2 still computes relative branches
    relative to rewritten_base_va=0 (identical relative displacements!), and caller
    only needs to patch rodata/label RIP-relative fixups after final VAs are placed.
    """
    # Collect unique PLT entries in deterministic order
    plt_entries: List[Tuple[str, int, int]] = []
    seen_plt_syms: Dict[str, int] = {}
    for va in sorted(analysis.plt_stubs.keys()):
        st = analysis.plt_stubs[va]
        if st.sym_name not in seen_plt_syms:
            slot_idx = 3 + len(plt_entries)
            seen_plt_syms[st.sym_name] = slot_idx
            plt_entries.append((st.sym_name, st.got_slot_va, slot_idx))

    # Also include address-taken imported functions in rel_dyn so they receive canonical PLT stubs
    for r in analysis.rel_dyn:
        if r.sym_name and r.r_type in (1, 6) and r.sym_name not in seen_plt_syms:
            if r.sym_idx < len(analysis.dynsyms):
                sym = analysis.dynsyms[r.sym_idx]
                if sym.st_shndx == 0 and sym.sym_type == 2 and sym.bind == 1:  # strictly STB_GLOBAL and STT_FUNC
                    slot_idx = 3 + len(plt_entries)
                    seen_plt_syms[r.sym_name] = slot_idx
                    plt_entries.append((r.sym_name, 0, slot_idx))

    fallback_funcs: List[Tuple[str, int]] = []
    fallback_va_set = set()
    if hybrid_mode != "static":
        for fva, cls in sorted(analysis.func_classifications.items()):
            if cls == "REQUIRES_DBT_FALLBACK":
                lbl = f"__nm_fb_{fva:08x}"
                fallback_funcs.append((lbl, fva))
                fallback_va_set.add(fva)

    dispatch_vec_sz = ((3 + len(plt_entries)) * 8 + 15) & ~15
    # rewritten_base_va is right after dispatch vector (aligned to 16B)
    # Callers pass exact rewritten_base_va; if 0, use placeholder 0 and recompute in pass 2.
    base_va = rewritten_base_va

    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
    md.detail = True

    code_buf = bytearray()
    orig_to_new: Dict[int, int] = {}
    branch_fixups: List[BranchFixup] = []
    rodata_fixups: List[Tuple[int, int, int]] = []
    label_fixups: List[Tuple[int, int, str]] = []
    jump_table_blobs: List[JumpTableBlob] = []
    rodata_off = 0

    def _emit_dispatch_call(slot_idx: int, cur_new_va: int) -> bytes:
        # leal .Lret(%rip), %r11d (7B); lea -4(%rsp), %rsp (5B); movl %r11d, (%rsp) (4B); jmp slot_idx (5B)
        # Note: after jmp (total 21B), .Lret is right at cur_new_va + 21!
        # Wait: leal disp(%rip), %r11d ends at cur_new_va + 7 -> disp = 21 - 7 = 14!
        buf = bytearray()
        buf.extend(_emit_lea_rip_to_r11d(14))
        buf.extend(b"\x48\x8d\x64\x24\xfc")
        buf.extend(b"\x44\x89\x1c\x24")
        jmp_off = len(code_buf) + len(buf) + 1
        buf.extend(b"\xe9\x00\x00\x00\x00")
        branch_fixups.append(
            BranchFixup(
                patch_off=jmp_off,
                next_insn_va=cur_new_va + len(buf),
                target_orig_va=slot_idx,
                target_kind="dispatch",
            )
        )
        return bytes(buf)

    def _emit_dispatch_jmp(slot_idx: int, cur_new_va: int) -> bytes:
        buf = bytearray(b"\xe9\x00\x00\x00\x00")
        jmp_off = len(code_buf) + 1
        branch_fixups.append(
            BranchFixup(
                patch_off=jmp_off,
                next_insn_va=cur_new_va + 5,
                target_orig_va=slot_idx,
                target_kind="dispatch",
            )
        )
        return bytes(buf)

    # Index jump table replacement spans
    jt_by_start: Dict[int, JumpTableInfo] = {
        jt.replace_start_va: jt for jt in analysis.jump_tables.values()
    }

    got_tracker = GotProvenanceTracker(
        analysis.got_va, analysis.pic_thunks, analysis.functions
    )

    # Collect known CFG/function/branch targets for resilient disassembly
    known_targets: Set[int] = set(analysis.functions)
    known_targets.update(analysis.reachable_insns.keys())
    known_targets.update(analysis.addr_taken_funcs)
    for jt in analysis.jump_tables.values():
        known_targets.update(jt.targets)
    if analysis.e_entry:
        known_targets.add(analysis.e_entry)
    if analysis.init_va:
        known_targets.add(analysis.init_va)
    if analysis.fini_va:
        known_targets.add(analysis.fini_va)
    for a in analysis.init_array + analysis.fini_array:
        known_targets.add(a)

    discovered_targets: Set[int] = set(known_targets)
    for cr in analysis.code_ranges:
        for insn in md.disasm(cr.data, cr.va):
            if insn.id in (
                cs_x86.X86_INS_JMP, cs_x86.X86_INS_CALL, cs_x86.X86_INS_LOOP,
                cs_x86.X86_INS_JCXZ, cs_x86.X86_INS_JECXZ,
            ) or insn.id in JCC_SHORT_MAP:
                if len(insn.operands) == 1 and insn.operands[0].type == cs_x86.X86_OP_IMM:
                    imm = insn.operands[0].imm & 0xFFFFFFFF
                    if any(c.va <= imm < c.va + len(c.data) for c in analysis.code_ranges):
                        discovered_targets.add(imm)

    def _check_and_emit_inline_data_after_uncond(insn_obj: capstone.CsInsn, cr_obj: CodeRange) -> Optional[int]:
        if insn_obj.id not in (
            cs_x86.X86_INS_JMP,
            cs_x86.X86_INS_RET,
            cs_x86.X86_INS_RETF,
            cs_x86.X86_INS_UD2,
            cs_x86.X86_INS_HLT,
        ):
            return None
        next_va = insn_obj.address + insn_obj.size
        if next_va < cr_obj.va + len(cr_obj.data) and next_va not in discovered_targets:
            ahead_t = [t for t in discovered_targets if next_va < t <= cr_obj.va + len(cr_obj.data)]
            if ahead_t:
                next_t = min(ahead_t)
                cur_off = next_va - cr_obj.va
                target_off = next_t - cr_obj.va
                while cur_off < target_off:
                    b_va = cr_obj.va + cur_off
                    orig_to_new[b_va] = base_va + len(code_buf)
                    code_buf.append(cr_obj.data[cur_off])
                    cur_off += 1
                return target_off
        return None

    # Build linear list of instructions per CodeRange using resilient chunked disassembly
    for cr in analysis.code_ranges:
        cr_off = 0
        cr_len = len(cr.data)
        while cr_off < cr_len:
            insns = list(md.disasm(cr.data[cr_off:], cr.va + cr_off))
            if insns:
                idx = 0
                broke_for_data = False
                while idx < len(insns):
                    insn = insns[idx]
                    va = insn.address
                    end_va = va + insn.size

                    # Check if insn straddles a discovered target (addr < t < addr + size)
                    straddled_t = None
                    for t in discovered_targets:
                        if va < t < end_va:
                            if straddled_t is None or t < straddled_t:
                                straddled_t = t

                    if straddled_t is not None:
                        # Do NOT emit that straddling instruction!
                        # Emit raw bytes from va up to straddled_t as inline data
                        cur_off = va - cr.va
                        target_off = straddled_t - cr.va
                        while cur_off < target_off:
                            b_va = cr.va + cur_off
                            orig_to_new[b_va] = base_va + len(code_buf)
                            code_buf.append(cr.data[cur_off])
                            cur_off += 1
                        cr_off = target_off
                        broke_for_data = True
                        break

                    cur_new_va = base_va + len(code_buf)
                    orig_to_new[va] = cur_new_va
                    next_insn = insns[idx + 1] if idx + 1 < len(insns) else None

                    # Dynamically update discovered_targets from branch instructions
                    if insn.id in (
                        cs_x86.X86_INS_JMP, cs_x86.X86_INS_CALL, cs_x86.X86_INS_LOOP,
                        cs_x86.X86_INS_JCXZ, cs_x86.X86_INS_JECXZ,
                    ) or insn.id in JCC_SHORT_MAP:
                        if len(insn.operands) == 1 and insn.operands[0].type == cs_x86.X86_OP_IMM:
                            imm = insn.operands[0].imm & 0xFFFFFFFF
                            if cr.va <= imm < cr.va + cr_len:
                                discovered_targets.add(imm)

                    # Check if this function entry is marked REQUIRES_DBT_FALLBACK
                    if va in fallback_va_set:
                        lbl = f"__nm_fb_{va:08x}"
                        patch_off = len(code_buf) + 1
                        code_buf.extend(b"\xe9\x00\x00\x00\x00")
                        label_fixups.append((patch_off, cur_new_va + 5, lbl))
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 1. Jump table replacement span
                    if va in jt_by_start:
                        jt = jt_by_start[va]
                        idx_r = REG32_IDX.get(jt.idx_reg, 0)
                        jt_blob = JumpTableBlob(rodata_offset=rodata_off, targets=list(jt.targets))
                        jump_table_blobs.append(jt_blob)
                        rodata_off += len(jt.targets) * 4

                        # leal jt(%rip), %r11d (7B)
                        # movl %idx32, %r10d    (3B: 41 89 <0xC2 | idx_r<<3>)
                        # addl (%r11,%r10,4), %r11d (4B: 47 03 1c 93)
                        # jmp *%r11             (3B: 41 ff e3)
                        patch_off = len(code_buf) + 3
                        code_buf.extend(_emit_lea_rip_to_r11d(0))
                        rodata_fixups.append((patch_off, cur_new_va + 7, jt_blob.rodata_offset))
                        code_buf.extend(bytes([0x41, 0x89, 0xC2 | (idx_r << 3)]))
                        code_buf.extend(b"\x47\x03\x1c\x93")
                        code_buf.extend(b"\x41\xff\xe3")

                        while idx < len(insns) and insns[idx].address < jt.replace_end_va:
                            if insns[idx].address not in orig_to_new:
                                orig_to_new[insns[idx].address] = cur_new_va
                            got_tracker.on_instruction(insns[idx])
                            idx += 1
                        continue

                    # 2. Inline `call $+5; pop %reg` PIC idiom
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
                        r_idx = REG32_IDX.get(pop_insn.operands[0].reg)
                        if r_idx is not None:
                            orig_pc = pop_insn.address
                            # leal (orig_pc - (cur_new_va + 6))(%rip), %reg32
                            disp = (orig_pc - (cur_new_va + 6)) & 0xFFFFFFFF
                            if disp >= 0x80000000:
                                disp -= 0x100000000
                            code_buf.extend(_emit_lea_rip_to_reg32(r_idx, disp))
                            orig_to_new[pop_insn.address] = cur_new_va
                            got_tracker.on_instruction(insn, pop_insn)
                            got_tracker.on_instruction(pop_insn)
                            idx += 2
                            continue

                    # 3. `call __x86.get_pc_thunk.<reg>`
                    if insn.id == cs_x86.X86_INS_CALL and len(insn.operands) == 1 and insn.operands[0].type == cs_x86.X86_OP_IMM:
                        tgt = insn.operands[0].imm & 0xFFFFFFFF
                        if tgt in analysis.pic_thunks:
                            r_idx = REG32_NAME_TO_IDX[analysis.pic_thunks[tgt]]
                            orig_ret = va + insn.size
                            disp = (orig_ret - (cur_new_va + 6)) & 0xFFFFFFFF
                            if disp >= 0x80000000:
                                disp -= 0x100000000
                            code_buf.extend(_emit_lea_rip_to_reg32(r_idx, disp))
                            got_tracker.on_instruction(insn, next_insn)
                            idx += 1
                            continue

                    # 4. GOTOFF function pointer: `lea disp(%got_reg), %dst_reg`
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
                            dst_idx = REG32_IDX[insn.operands[0].reg]
                            patch_off = len(code_buf) + 2
                            code_buf.extend(_emit_lea_rip_to_reg32(dst_idx, 0))
                            label_fixups.append((patch_off, cur_new_va + 6, f"__nm_dual_{cand:08x}"))
                            got_tracker.on_instruction(insn, next_insn)
                            idx += 1
                            continue

                    # 5. Unreachable `hlt` (e.g. crt1.o fallthrough after `call __libc_start_main`)
                    if insn.id == cs_x86.X86_INS_HLT:
                        if va in analysis.reachable_insns:
                            raise PatcherError(f"reachable 'hlt' at 0x{va:08x}")
                        code_buf.append(0xCC)
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 6. `endbr32` -> `endbr64`
                    if insn.id == cs_x86.X86_INS_ENDBR32:
                        code_buf.extend(b"\xf3\x0f\x1e\xfa")
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 7. Multi-byte NOPs
                    if _is_gcc_multibyte_nop(insn):
                        code_buf.append(0x90)
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 8. `int 0x80` / `sysenter`
                    if (insn.id == cs_x86.X86_INS_INT and len(insn.bytes) >= 2 and insn.bytes[1] == 0x80) or (
                        insn.id == cs_x86.X86_INS_SYSENTER
                    ):
                        code_buf.extend(_emit_dispatch_call(1, cur_new_va))
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 9. `0x40..0x4F` `inc`/`dec r32` or prefixed (e.g. 0x66) `inc`/`dec r16`
                    if (
                        (len(insn.bytes) == 1 and 0x40 <= insn.bytes[0] <= 0x4F)
                        or (
                            insn.id in (cs_x86.X86_INS_INC, cs_x86.X86_INS_DEC)
                            and len(insn.bytes) >= 1
                            and 0x40 <= insn.bytes[-1] <= 0x4F
                        )
                    ):
                        b_op = insn.bytes[-1]
                        modrm = (0xC0 + (b_op - 0x40)) if b_op <= 0x47 else (0xC8 + (b_op - 0x48))
                        prefixes = bytes(insn.bytes[:-1])
                        code_buf.extend(prefixes + bytes([0xFF, modrm]))
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 10. `pusha` / `popa`
                    if insn.id == cs_x86.X86_INS_PUSHAL:
                        # movl %esp, %r11d; lea -32(%rsp), %rsp; save eax,ecx,edx,ebx,orig_esp,ebp,esi,edi
                        code_buf.extend(b"\x41\x89\xe3")              # movl %esp, %r11d
                        code_buf.extend(b"\x48\x8d\x64\x24\xe0")      # lea -32(%rsp), %rsp
                        # offsets: +28=eax, +24=ecx, +20=edx, +16=ebx, +12=r11d, +8=ebp, +4=esi, +0=edi
                        for reg_i, off_s in ((0, 28), (1, 24), (2, 20), (3, 16), (5, 8), (6, 4), (7, 0)):
                            code_buf.extend(bytes([0x89, 0x44 | (reg_i << 3), 0x24, off_s]))
                        code_buf.extend(b"\x44\x89\x5c\x24\x0c")      # movl %r11d, 12(%rsp)
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    if insn.id == cs_x86.X86_INS_POPAL:
                        for reg_i, off_s in ((7, 0), (6, 4), (5, 8), (3, 16), (2, 20), (1, 24), (0, 28)):
                            code_buf.extend(bytes([0x8B, 0x44 | (reg_i << 3), 0x24, off_s]))
                        code_buf.extend(b"\x48\x8d\x64\x24\x20")      # lea 32(%rsp), %rsp
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 11. `push` variants
                    if insn.id == cs_x86.X86_INS_PUSH and len(insn.operands) == 1:
                        op = insn.operands[0]
                        if op.type == cs_x86.X86_OP_REG and op.reg in REG32_IDX:
                            code_buf.extend(_emit_push_reg32(REG32_IDX[op.reg]))
                        elif op.type == cs_x86.X86_OP_IMM:
                            imm = op.imm & 0xFFFFFFFF
                            if analysis.e_type == 2 and imm in analysis.addr_taken_funcs:
                                # ET_EXEC push $func_va -> leal dual(%rip), %r11d; push %r11d
                                patch_off = len(code_buf) + 3
                                code_buf.extend(_emit_lea_rip_to_r11d(0))
                                label_fixups.append((patch_off, cur_new_va + 7, f"__nm_dual_{imm:08x}"))
                                code_buf.extend(b"\x48\x8d\x64\x24\xfc\x44\x89\x1c\x24")
                            else:
                                if imm >= 0x80000000:
                                    imm -= 0x100000000
                                code_buf.extend(_emit_push_imm32(imm))
                        elif op.type == cs_x86.X86_OP_MEM:
                            code_buf.extend(_rewrite_rm32_mov_r11d(insn, store_to_mem=False))
                            code_buf.extend(b"\x48\x8d\x64\x24\xfc\x44\x89\x1c\x24")
                        else:
                            raise PatcherError(
                                f"unsupported push operand at 0x{va:08x}: {insn.mnemonic} {insn.op_str}"
                            )
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 12. `pop` variants
                    if insn.id == cs_x86.X86_INS_POP and len(insn.operands) == 1:
                        op = insn.operands[0]
                        if op.type == cs_x86.X86_OP_REG and op.reg in REG32_IDX:
                            code_buf.extend(_emit_pop_reg32(REG32_IDX[op.reg]))
                        elif op.type == cs_x86.X86_OP_MEM:
                            code_buf.extend(b"\x44\x8b\x1c\x24\x48\x8d\x64\x24\x04")
                            code_buf.extend(_rewrite_rm32_mov_r11d(insn, store_to_mem=True))
                        else:
                            raise PatcherError(
                                f"unsupported pop operand at 0x{va:08x}: {insn.mnemonic} {insn.op_str}"
                            )
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 13. `pushf` / `popf`
                    if insn.id in (cs_x86.X86_INS_PUSHF, cs_x86.X86_INS_PUSHFD):
                        code_buf.extend(b"\x9c\x41\x5b\x48\x8d\x64\x24\xfc\x44\x89\x1c\x24")
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue
                    if insn.id in (cs_x86.X86_INS_POPF, cs_x86.X86_INS_POPFD):
                        code_buf.extend(b"\x44\x8b\x1c\x24\x48\x8d\x64\x24\x04\x41\x53\x9d")
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 14. `leave` / `enter`
                    if insn.id == cs_x86.X86_INS_LEAVE:
                        code_buf.extend(b"\x89\xec\x8b\x2c\x24\x48\x8d\x64\x24\x04")
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue
                    if insn.id == cs_x86.X86_INS_ENTER and len(insn.operands) == 2:
                        frame_sz = int(insn.operands[0].imm) & 0xFFFF
                        code_buf.extend(_emit_push_reg32(5))  # push ebp
                        code_buf.extend(b"\x89\xe5")          # mov ebp, esp
                        if frame_sz:
                            code_buf.extend(b"\x48\x8d\xa4\x24" + struct.pack("<i", -frame_sz))
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 15. `ret` / `ret imm16`
                    if insn.id == cs_x86.X86_INS_RET:
                        adj = 4
                        if len(insn.operands) == 1 and insn.operands[0].type == cs_x86.X86_OP_IMM:
                            adj += int(insn.operands[0].imm) & 0xFFFF
                        code_buf.extend(b"\x44\x8b\x1c\x24")  # movl (%rsp), %r11d
                        if adj <= 127:
                            code_buf.extend(bytes([0x48, 0x8D, 0x64, 0x24, adj]))
                        else:
                            code_buf.extend(b"\x48\x8d\xa4\x24" + struct.pack("<i", adj))
                        code_buf.extend(b"\x41\xff\xe3")      # jmp *%r11
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 16. `call`
                    if insn.id == cs_x86.X86_INS_CALL and len(insn.operands) == 1:
                        op = insn.operands[0]
                        if op.type == cs_x86.X86_OP_IMM:
                            tgt = op.imm & 0xFFFFFFFF
                            if tgt in analysis.plt_stubs:
                                slot_idx = seen_plt_syms[analysis.plt_stubs[tgt].sym_name]
                                code_buf.extend(_emit_dispatch_call(slot_idx, cur_new_va))
                            else:
                                # Direct internal call:
                                # leal .Lret(%rip), %r11d (7B); lea -4(%rsp), %rsp (5B); movl %r11d, (%rsp) (4B); jmp tgt (5B)
                                code_buf.extend(_emit_lea_rip_to_r11d(14))
                                code_buf.extend(b"\x48\x8d\x64\x24\xfc\x44\x89\x1c\x24")
                                jmp_off = len(code_buf) + 1
                                code_buf.extend(b"\xe9\x00\x00\x00\x00")
                                branch_fixups.append(
                                    BranchFixup(
                                        patch_off=jmp_off,
                                        next_insn_va=cur_new_va + 21,
                                        target_orig_va=tgt,
                                        target_kind="orig",
                                    )
                                )
                        else:
                            # Indirect call `call *r/m32`
                            # movl r/m32, %r11d; leal .Lret(%rip), %r10d; lea -4(%rsp), %rsp; movl %r10d, (%rsp); jmp __nm_icall
                            code_buf.extend(_rewrite_rm32_mov_r11d(insn, store_to_mem=False))
                            sub_start = len(code_buf)
                            # Remaining: 7 + 5 + 4 + 5 = 21 bytes; disp for leal is 21 - 7 = 14
                            code_buf.extend(_emit_lea_rip_to_r10d(14))
                            code_buf.extend(b"\x48\x8d\x64\x24\xfc\x44\x89\x14\x24")
                            jmp_off = len(code_buf) + 1
                            code_buf.extend(b"\xe9\x00\x00\x00\x00")
                            branch_fixups.append(
                                BranchFixup(
                                    patch_off=jmp_off,
                                    next_insn_va=base_va + sub_start + 21,
                                    target_orig_va=0,
                                    target_kind="dispatch",
                                )
                            )
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 17. `jmp`
                    if insn.id == cs_x86.X86_INS_JMP and len(insn.operands) == 1:
                        op = insn.operands[0]
                        if op.type == cs_x86.X86_OP_IMM:
                            tgt = op.imm & 0xFFFFFFFF
                            if tgt in analysis.plt_stubs:
                                slot_idx = seen_plt_syms[analysis.plt_stubs[tgt].sym_name]
                                code_buf.extend(_emit_dispatch_jmp(slot_idx, cur_new_va))
                            else:
                                jmp_off = len(code_buf) + 1
                                code_buf.extend(b"\xe9\x00\x00\x00\x00")
                                branch_fixups.append(
                                    BranchFixup(
                                        patch_off=jmp_off,
                                        next_insn_va=cur_new_va + 5,
                                        target_orig_va=tgt,
                                        target_kind="orig",
                                    )
                                )
                        else:
                            # Tail indirect jump `jmp *r/m32` (not a jump table)
                            if insn.address in analysis.plt_stubs:
                                slot_idx = seen_plt_syms[analysis.plt_stubs[insn.address].sym_name]
                                code_buf.extend(_emit_dispatch_jmp(slot_idx, cur_new_va))
                            else:
                                code_buf.extend(_rewrite_rm32_mov_r11d(insn, store_to_mem=False))
                                code_buf.extend(_emit_dispatch_jmp(2, base_va + len(code_buf)))
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 18. Conditional branches `jcc`
                    if insn.id in JCC_SHORT_MAP and len(insn.operands) == 1 and insn.operands[0].type == cs_x86.X86_OP_IMM:
                        cc = JCC_SHORT_MAP[insn.id]
                        tgt = insn.operands[0].imm & 0xFFFFFFFF
                        patch_off = len(code_buf) + 2
                        code_buf.extend(bytes([0x0F, 0x80 + cc, 0, 0, 0, 0]))
                        branch_fixups.append(
                            BranchFixup(
                                patch_off=patch_off,
                                next_insn_va=cur_new_va + 6,
                                target_orig_va=tgt,
                                target_kind="orig",
                            )
                        )
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    if insn.id in (cs_x86.X86_INS_JCXZ, cs_x86.X86_INS_JECXZ):
                        tgt = insn.operands[0].imm & 0xFFFFFFFF
                        # Zero EFLAGS modified:
                        # For jecxz (32-bit count %ecx): 67 e3 02 eb 05 e9 <rel32> (10 bytes)
                        #   67 e3 02: jecxz +2 -> jump over 'eb 05' (2 bytes) directly to 'e9 <rel32>'
                        #   eb 05:    jmp +5   -> jump over the 5-byte 'e9 <rel32>' to next instruction
                        #   e9 rel32: jmp rel32 -> jump to target VA
                        # For jcxz (16-bit count %cx): 66 67 e3 02 eb 05 e9 <rel32> (11 bytes)
                        if insn.id == cs_x86.X86_INS_JECXZ:
                            code_buf.extend(b"\x67\xe3\x02\xeb\x05\xe9\x00\x00\x00\x00")
                            next_va = cur_new_va + 10
                        else:
                            code_buf.extend(b"\x66\x67\xe3\x02\xeb\x05\xe9\x00\x00\x00\x00")
                            next_va = cur_new_va + 11
                        branch_fixups.append(
                            BranchFixup(
                                patch_off=len(code_buf) - 4,
                                next_insn_va=next_va,
                                target_orig_va=tgt,
                                target_kind="orig",
                            )
                        )
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    if insn.id == cs_x86.X86_INS_LOOP:
                        tgt = insn.operands[0].imm & 0xFFFFFFFF
                        # leal -1(%ecx), %ecx (3B: 67 8d 49 ff); jecxz +5 (2B: 67 e3 05); jmp rel32 (5B)
                        code_buf.extend(b"\x67\x8d\x49\xff\x67\xe3\x05\xe9\x00\x00\x00\x00")
                        branch_fixups.append(
                            BranchFixup(
                                patch_off=len(code_buf) - 4,
                                next_insn_va=cur_new_va + 12,
                                target_orig_va=tgt,
                                target_kind="orig",
                            )
                        )
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 19. ET_EXEC `mov $func_va, %reg32` (B8+r imm32)
                    if (
                        analysis.e_type == 2
                        and insn.id == cs_x86.X86_INS_MOV
                        and len(insn.operands) == 2
                        and insn.operands[0].type == cs_x86.X86_OP_REG
                        and insn.operands[0].reg in REG32_IDX
                        and insn.operands[1].type == cs_x86.X86_OP_IMM
                        and (insn.operands[1].imm & 0xFFFFFFFF) in analysis.addr_taken_funcs
                    ):
                        dst_idx = REG32_IDX[insn.operands[0].reg]
                        fva = insn.operands[1].imm & 0xFFFFFFFF
                        patch_off = len(code_buf) + 2
                        code_buf.extend(_emit_lea_rip_to_reg32(dst_idx, 0))
                        label_fixups.append((patch_off, cur_new_va + 6, f"__nm_dual_{fva:08x}"))
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 19b. ET_EXEC `movl $func_va, mem` (C7 /0 imm32)
                    if (
                        analysis.e_type == 2
                        and insn.id == cs_x86.X86_INS_MOV
                        and len(insn.operands) == 2
                        and insn.operands[0].type == cs_x86.X86_OP_MEM
                        and insn.operands[1].type == cs_x86.X86_OP_IMM
                        and (insn.operands[1].imm & 0xFFFFFFFF) in analysis.addr_taken_funcs
                    ):
                        fva = insn.operands[1].imm & 0xFFFFFFFF
                        patch_off = len(code_buf) + 3
                        code_buf.extend(_emit_lea_rip_to_r11d(0))
                        label_fixups.append((patch_off, cur_new_va + 7, f"__nm_dual_{fva:08x}"))
                        code_buf.extend(_rewrite_rm32_mov_r11d(insn, store_to_mem=True, has_imm32=True))
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 20. Segment-prefixed `%gs:` instructions
                    if insn.prefix[1] == 0x65:
                        code_buf.extend(_rewrite_gs_insn(insn))
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 21. `0xA0..0xA3` moffs32 without %gs:
                    raw = bytes(insn.bytes)
                    op_scan = 0
                    while op_scan < len(raw) and raw[op_scan] in (0x66, 0x67, 0xF0, 0xF2, 0xF3):
                        op_scan += 1
                    if op_scan < len(raw) and raw[op_scan] in (0xA0, 0xA1, 0xA2, 0xA3):
                        code_buf.extend(_rewrite_moffs(insn))
                        got_tracker.on_instruction(insn, next_insn)
                        new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                        if new_off is not None:
                            cr_off = new_off
                            broke_for_data = True
                            break
                        idx += 1
                        continue

                    # 22. General instructions: check if memory operand or string instruction needs 0x67 / SIB 0x25
                    has_mem = any(op.type == cs_x86.X86_OP_MEM for op in insn.operands)
                    is_string_op = insn.id in (
                        cs_x86.X86_INS_MOVSB, cs_x86.X86_INS_MOVSW, cs_x86.X86_INS_MOVSD,
                        cs_x86.X86_INS_STOSB, cs_x86.X86_INS_STOSW, cs_x86.X86_INS_STOSD,
                        cs_x86.X86_INS_LODSB, cs_x86.X86_INS_LODSW, cs_x86.X86_INS_LODSD,
                        cs_x86.X86_INS_CMPSB, cs_x86.X86_INS_CMPSW, cs_x86.X86_INS_CMPSD,
                        cs_x86.X86_INS_SCASB, cs_x86.X86_INS_SCASW, cs_x86.X86_INS_SCASD,
                    )
                    modrm_off = insn.encoding.modrm_offset
                    code_buf.extend(
                        _fix_modrm_disp32_and_addr32(
                            raw,
                            modrm_off,
                            is_gs=False,
                            add_addr32=(has_mem or is_string_op),
                        )
                    )
                    got_tracker.on_instruction(insn, next_insn)
                    new_off = _check_and_emit_inline_data_after_uncond(insn, cr)
                    if new_off is not None:
                        cr_off = new_off
                        broke_for_data = True
                        break
                    idx += 1

                if not broke_for_data:
                    last_insn = insns[-1]
                    cr_off = max(cr_off + 1, last_insn.address + last_insn.size - cr.va)
            else:
                # md.disasm could not decode at cr_off; advance past undecodable bytes
                curr_va = cr.va + cr_off
                ahead = [v for v in sorted(discovered_targets) if curr_va < v < cr.va + cr_len]
                if ahead:
                    target_off = ahead[0] - cr.va
                    while cr_off < target_off:
                        byte_va = cr.va + cr_off
                        cur_new_va = base_va + len(code_buf)
                        orig_to_new[byte_va] = cur_new_va
                        code_buf.append(cr.data[cr_off])
                        cr_off += 1
                else:
                    while cr_off < cr_len:
                        curr_va = cr.va + cr_off
                        test_insn = next(md.disasm(cr.data[cr_off:], curr_va, count=1), None)
                        if test_insn is not None:
                            break
                        cur_new_va = base_va + len(code_buf)
                        orig_to_new[curr_va] = cur_new_va
                        code_buf.append(cr.data[cr_off])
                        cr_off += 1

    # Pass 2: resolve internal branches & dispatch vector jumps
    for bf in branch_fixups:
        if bf.target_kind == "dispatch":
            target_va = rewritten_base_va - dispatch_vec_sz + bf.target_orig_va * 8
        else:
            if bf.target_orig_va not in orig_to_new:
                raise PatcherError(
                    f"branch target 0x{bf.target_orig_va:08x} not found in rewritten code map"
                )
            target_va = orig_to_new[bf.target_orig_va]
        rel = target_va - bf.next_insn_va
        struct.pack_into("<i", code_buf, bf.patch_off, rel)

    if verbose:
        import sys
        sys.stderr.write(
            f"[newminhh:rewriter] rewritten {len(orig_to_new)} insns -> {len(code_buf)} bytes, "
            f"{len(jump_table_blobs)} jump tables, {len(plt_entries)} PLTs\n"
        )

    return RewrittenOutput(
        code_bytes=code_buf,
        rewritten_base_va=base_va,
        orig_to_new=orig_to_new,
        jump_table_blobs=jump_table_blobs,
        rodata_fixups=rodata_fixups,
        label_fixups=label_fixups,
        plt_entries=plt_entries,
        fallback_funcs=fallback_funcs,
        rodata_jump_tables_size=rodata_off,
    )
