"""Strict static purity gates for Level 3A per-function freestanding -mx32 boosting."""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    import capstone
except ImportError:
    capstone = None  # type: ignore

STACK_REGS = frozenset({"esp", "ebp"})
ENTRY_REGS = ("eax", "ecx", "edx")
CALLEE_SAVED_REGS = frozenset({"ebx", "esi", "edi", "ebp"})
TRACKED_GPRS = frozenset({"eax", "ecx", "edx", "ebx", "esi", "edi"})
MAX_SCALAR_PARAMS = 6

# Map every 8/16/32-bit GPR name to its 32-bit parent register.
_GPR_PARENT: Dict[str, str] = {
    "eax": "eax", "ax": "eax", "al": "eax", "ah": "eax",
    "ecx": "ecx", "cx": "ecx", "cl": "ecx", "ch": "ecx",
    "edx": "edx", "dx": "edx", "dl": "edx", "dh": "edx",
    "ebx": "ebx", "bx": "ebx", "bl": "ebx", "bh": "ebx",
    "esi": "esi", "si": "esi", "sil": "esi",
    "edi": "edi", "di": "edi", "dil": "edi",
    "ebp": "ebp", "bp": "ebp", "bpl": "ebp",
    "esp": "esp", "sp": "esp", "spl": "esp",
}

_GPR_SUBREGS: Dict[str, Tuple[str, ...]] = {
    "eax": ("eax", "ax", "al", "ah"),
    "ecx": ("ecx", "cx", "cl", "ch"),
    "edx": ("edx", "dx", "dl", "dh"),
    "ebx": ("ebx", "bx", "bl", "bh"),
    "esi": ("esi", "si", "sil"),
    "edi": ("edi", "di", "dil"),
    "ebp": ("ebp", "bp", "bpl"),
    "esp": ("esp", "sp", "spl"),
}

_XMM_REGS = frozenset(f"xmm{i}" for i in range(8))
_IGNORED_SYS_REGS = frozenset({"esp", "sp", "spl", "eflags", "flags", "eip", "ip", "fpsw", "cs", "ds", "es", "ss"})

FORBIDDEN_PRIVILEGE_MNEMONICS = frozenset({
    "int", "int1", "into", "syscall", "sysenter", "sysret", "sysexit",
    "hlt", "in", "out", "ins", "insb", "insw", "insd",
    "outs", "outsb", "outsw", "outsd", "rdmsr", "wrmsr", "rdpmc",
    "cpuid", "rdtsc", "rdtscp", "xgetbv", "xsetbv",
    "cli", "sti", "lidt", "lgdt", "lldt", "ltr", "sidt", "sgdt", "sldt", "str",
    "lmsw", "smsw", "clts", "invd", "wbinvd", "invlpg", "swapgs",
    "iret", "iretd", "iretq", "ud0", "ud1", "ud2",
})

_NOP_MNEMONICS = frozenset({"nop", "fnop", "endbr32", "endbr64"})
_SELF_ZERO_2OP = frozenset({"xor", "sub", "pxor", "xorps", "xorpd"})
_SELF_ZERO_3OP = frozenset({"vpxor", "vxorps", "vxorpd"})

_X87_PUSH_MNEMONICS = frozenset({
    "fld", "fild", "fbld", "fldz", "fld1", "fldpi", "fldl2e", "fldl2t", "fldlg2", "fldln2",
})
_X87_POP1_MNEMONICS = frozenset({
    "fstp", "fistp", "fbstp", "fcomp", "fucomp", "faddp", "fsubp", "fsubrp", "fmulp", "fdivp", "fdivrp",
})
_X87_POP2_MNEMONICS = frozenset({"fcompp", "fucompp"})
_X87_READ_MNEMONICS = frozenset({
    "fst", "fist", "fadd", "fsub", "fsubr", "fmul", "fdiv", "fdivr",
    "fcom", "fucom", "fchs", "fabs", "fsqrt", "frndint", "fxam", "ftst",
    "fsin", "fcos", "fsincos", "fptan", "fpatan", "f2xm1", "fyl2x", "fyl2xp1",
})

_ST_RE = re.compile(r"^st(?:\((\d+)\)|(\d+))?$")


class ScalarType(enum.Enum):
    """Scalar types used by the Level 3A freestanding boosting pipeline."""

    I32 = "i32"
    U32 = "u32"
    I64 = "i64"
    U64 = "u64"
    F32 = "f32"
    F64 = "f64"
    VOID = "void"
    PTR_CONST_BYTES = "ptr_const_bytes"


@dataclass
class FunctionGateCandidate:
    """A decompiled function candidate evaluated by the purity gates."""

    name: str
    va: int
    size: int
    code: bytes
    ret_type: str = ScalarType.I32.value
    arg_types: List[str] = field(default_factory=list)
    stack_frame_bytes: int = 0


@dataclass
class GateVerdict:
    """Strict purity verdict for a function candidate."""

    passed: bool = False
    reasons: List[str] = field(default_factory=list)
    ret_type: str = ScalarType.I32.value
    arg_types: List[str] = field(default_factory=list)
    stack_frame_bytes: int = 0
    pure_callees: List[str] = field(default_factory=list)


def _cs_const(name: str, default: Any) -> Any:
    return getattr(capstone, name, default) if capstone is not None else default


OP_REG = _cs_const("X86_OP_REG", 1)
OP_IMM = _cs_const("X86_OP_IMM", 2)
OP_MEM = _cs_const("X86_OP_MEM", 3)

_READ_MASK = (
    (_cs_const("CS_AC_READ", 1) or 1)
    | (_cs_const("X86_OP_ACCESS_READ", 0) or 0)
)
_WRITE_MASK = (
    (_cs_const("CS_AC_WRITE", 2) or 2)
    | (_cs_const("X86_OP_ACCESS_WRITE", 0) or 0)
)


def _reg_name(insn: Any, reg_id: Any) -> str:
    if not reg_id:
        return ""
    try:
        return str(insn.reg_name(reg_id) or "").lower()
    except Exception:
        return ""


def _op_reg_name(insn: Any, op: Any) -> str:
    if getattr(op, "type", None) != OP_REG:
        return ""
    return _reg_name(insn, getattr(op, "reg", 0))


def _op_imm(op: Any) -> Optional[int]:
    if getattr(op, "type", None) != OP_IMM:
        return None
    try:
        return int(op.imm)
    except Exception:
        return None


def _op_mem_fields(insn: Any, op: Any) -> Tuple[str, str, int, str]:
    mem = getattr(op, "mem", None)
    if mem is None:
        return "", "", 0, ""
    base = _reg_name(insn, getattr(mem, "base", 0))
    index = _reg_name(insn, getattr(mem, "index", 0))
    disp = int(getattr(mem, "disp", 0) or 0)
    seg = _reg_name(insn, getattr(mem, "segment", 0))
    return base, index, disp, seg


def _is_read_op(op: Any) -> bool:
    acc = int(getattr(op, "access", 0) or 0)
    return True if acc == 0 else bool(acc & _READ_MASK)


def _is_write_op(op: Any) -> bool:
    acc = int(getattr(op, "access", 0) or 0)
    return True if acc == 0 else bool(acc & _WRITE_MASK)


def _self_zero_reg(insn: Any, mn: str, ops: List[Any]) -> Optional[str]:
    """Return destination register if instruction is a self-zeroing idiom."""
    if mn in _SELF_ZERO_2OP and len(ops) >= 2:
        r0, r1 = _op_reg_name(insn, ops[0]), _op_reg_name(insn, ops[1])
        if r0 and r0 == r1:
            return r0
    if mn in _SELF_ZERO_3OP and len(ops) >= 3:
        r0, r1, r2 = (
            _op_reg_name(insn, ops[0]),
            _op_reg_name(insn, ops[1]),
            _op_reg_name(insn, ops[2]),
        )
        if r0 and r1 and r1 == r2:
            return r0
    return None


def _mark_written(written: Set[str], reg: str) -> None:
    """Record a register write; 32-bit GPR writes also initialize sub-registers."""
    if not reg:
        return
    written.add(reg)
    if reg in _GPR_SUBREGS:
        written.update(_GPR_SUBREGS[reg])


def _is_reg_written(written: Set[str], reg: str) -> bool:
    return reg in written


def _flags_has_pure(flags: Any) -> bool:
    if flags is None:
        return False
    if isinstance(flags, bytes):
        flags = flags.decode("utf-8", "replace")
    if isinstance(flags, str):
        return "PURE" in flags.upper()
    if isinstance(flags, Iterable):
        return "PURE" in {str(f).upper() for f in flags}
    return False


def _kind_is_native_typed(kind: Any) -> bool:
    if isinstance(kind, bytes):
        kind = kind.decode("utf-8", "replace")
    return isinstance(kind, str) and kind.strip().upper() == "NATIVE_TYPED"


def _import_info(known_imports: Dict[int, Any], target: int) -> Optional[Tuple[str, Any, Any]]:
    info = known_imports.get(target)
    if info is None:
        return None
    if isinstance(info, dict):
        return str(info.get("name") or ""), info.get("flags"), info.get("kind")
    values = list(info)
    name = str(values[0]) if len(values) > 0 else ""
    flags = values[1] if len(values) > 1 else ""
    kind = values[2] if len(values) > 2 else ""
    return name, flags, kind


def _parse_st_index(reg: str) -> Optional[int]:
    m = _ST_RE.match(reg)
    if not m:
        return None
    idx = m.group(1) or m.group(2)
    return int(idx) if idx is not None else 0


def _infer_arg_slot_type(mn: str, op_size: int) -> str:
    if mn in ("movss", "addss", "subss", "mulss", "divss", "ucomiss", "comiss"):
        return ScalarType.F32.value
    if mn in ("movsd", "addsd", "subsd", "mulsd", "divsd", "ucomisd", "comisd"):
        return ScalarType.F64.value
    if mn.startswith("f") and op_size == 8:
        return ScalarType.F64.value
    if mn.startswith("f") and op_size == 4:
        return ScalarType.F32.value
    if op_size == 8:
        return ScalarType.I64.value
    return ScalarType.I32.value


def _is_branch_or_call(mn: str) -> bool:
    return mn == "call" or mn.startswith("j") or mn.startswith("loop")


def _check_segment_override(insn: Any, addr: int, op_str: str, reasons: List[str]) -> None:
    prefixes = getattr(insn, "prefix", ()) or ()
    if len(prefixes) > 1 and prefixes[1] in (0x64, 0x65):
        seg = "fs" if prefixes[1] == 0x64 else "gs"
        reasons.append(f"gates: fs/gs segment override at {hex(addr)}: {seg}")
        return
    for seg_name in ("fs", "gs"):
        if f"{seg_name}:" in op_str:
            reasons.append(f"gates: fs/gs segment override at {hex(addr)}: {seg_name}")


def _check_jump_targets(
    addr: int,
    mn: str,
    ops: List[Any],
    func_va: int,
    func_end: int,
    resolved: Any,
    reasons: List[str],
) -> None:
    if mn == "jmp":
        imm_vals = [v for v in (_op_imm(op) for op in ops) if v is not None]
        if imm_vals:
            if not any(func_va <= v < func_end for v in imm_vals):
                reasons.append(f"gates: direct jump outside function at {hex(addr)} -> {hex(imm_vals[0])}")
            return
        if isinstance(resolved, dict):
            targets = resolved.get(addr)
            if targets is None or not all(func_va <= int(t) < func_end for t in targets):
                reasons.append(f"gates: indirect jump at {hex(addr)} not resolved inside function")
        elif addr not in resolved:
            reasons.append(f"gates: indirect jump at {hex(addr)} not in resolved_jump_tables")
    elif mn.startswith("j") or mn.startswith("loop"):
        for op in ops:
            tgt = _op_imm(op)
            if tgt is not None and not (func_va <= tgt < func_end):
                reasons.append(f"gates: direct jump outside function at {hex(addr)} ({mn}) -> {hex(tgt)}")


def evaluate_purity_gate(
    func_name: str,
    func_va: int,
    func_bytes: bytes,
    *,
    seg_ranges: Optional[List[Tuple[int, int]]] = None,
    known_imports: Optional[Dict[int, Tuple[str, str, str]]] = None,
    resolved_jump_tables: Optional[Set[int]] = None,
    max_stack_frame: int = 4096,
) -> GateVerdict:
    """Strictly evaluate all 10 static purity gates for a 32-bit x86 function.

    Disassembles ``func_bytes`` at ``func_va`` with capstone (CS_ARCH_X86, CS_MODE_32)
    and fails closed on any unsupported instruction, PIC/GOT sequence, external/indirect
    call, non-stack memory access, or entry register read before write.
    """
    if capstone is None:
        return GateVerdict(
            passed=False,
            reasons=["gates: capstone is unavailable (fail-closed)"],
            ret_type=ScalarType.I32.value,
        )

    seg_list = list(seg_ranges or [])
    imports = dict(known_imports or {})
    resolved = resolved_jump_tables if resolved_jump_tables is not None else set()
    code = bytes(func_bytes or b"")
    func_end = func_va + len(code)

    if not code:
        return GateVerdict(passed=False, reasons=["gates: empty function bytes"])

    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
    md.detail = True
    try:
        insns = list(md.disasm(code, func_va))
    except Exception as exc:
        return GateVerdict(passed=False, reasons=[f"gates: disassembly failed: {exc!r}"])

    if not insns:
        return GateVerdict(passed=False, reasons=["gates: function has no decodable instructions"])

    reasons: List[str] = []
    pure_callees: List[str] = []

    # ---- Gate 1: complete decodable bytes (ignoring trailing nop/int3 padding) ----
    covered = bytearray(len(code))
    for insn in insns:
        off = int(insn.address) - func_va
        end = min(off + int(insn.size), len(code))
        for i in range(max(0, off), end):
            covered[i] = 1

    first_uncovered = next((i for i, flag in enumerate(covered) if not flag), None)
    if first_uncovered is not None:
        trailing = code[first_uncovered:]
        if not all(b in (0x90, 0xCC) for b in trailing):
            reasons.append(f"gates: undecodable byte at {hex(func_va + first_uncovered)}")

    # Find last non-padding instruction index so trailing int3 padding can be ignored.
    last_body_idx = -1
    for idx, insn in enumerate(insns):
        mn = str(insn.mnemonic or "").lower()
        if mn not in _NOP_MNEMONICS and mn != "int3":
            last_body_idx = idx

    written: Set[str] = set()
    escaped_bases: Set[str] = set()
    arg_slots: Dict[int, str] = {}
    x87_depth = 0
    x87_last_size = 4
    saw_ret = False
    has_frame_ptr = False
    prev_was_push_ebp = False
    esp_adjust = 0
    local_frame = 0
    max_frame = 0
    edx_live_out = False

    for idx, insn in enumerate(insns):
        addr = int(insn.address)
        size = int(insn.size)
        mn = str(insn.mnemonic or "").lower()
        op_str = str(insn.op_str or "").lower()
        ops = list(getattr(insn, "operands", None) or [])

        if mn in _NOP_MNEMONICS:
            prev_was_push_ebp = False
            continue

        if mn == "int3":
            if idx <= last_body_idx:
                reasons.append(f"gates: embedded int3 trap at {hex(addr)}")
            prev_was_push_ebp = False
            continue

        # ---- Gate 2: no syscalls / privileged ops ----
        if mn in FORBIDDEN_PRIVILEGE_MNEMONICS:
            reasons.append(f"gates: forbidden privileged/syscall instruction at {hex(addr)}: {mn}")

        # ---- Gate 3: no %fs:/%gs: segment overrides ----
        _check_segment_override(insn, addr, op_str, reasons)

        # ---- Gate 4a: no absolute memory operands ----
        for op in ops:
            if getattr(op, "type", None) != OP_MEM:
                continue
            base, index, disp, seg = _op_mem_fields(insn, op)
            if seg in ("fs", "gs"):
                reasons.append(f"gates: fs/gs segment override at {hex(addr)}: {seg}")
            if not base and not index:
                reasons.append(f"gates: absolute memory operand at {hex(addr)}: disp={hex(disp)}")

        # ---- Gate 4c: non-control-flow immediates inside seg_ranges (va >= 0x1000) ----
        if not _is_branch_or_call(mn):
            for op in ops:
                val = _op_imm(op)
                if val is None or val < 0x1000:
                    continue
                for seg_start, seg_end in seg_list:
                    if seg_start <= val < seg_end:
                        reasons.append(
                            f"gates: immediate {hex(val)} at {hex(addr)} falls inside segment "
                            f"[{hex(seg_start)}, {hex(seg_end)})"
                        )
                        break

        # ---- Gate 4b + Gate 5: calls ----
        if mn == "call":
            call_imms = [v for v in (_op_imm(op) for op in ops) if v is not None]
            if not call_imms:
                reasons.append(f"gates: indirect call at {hex(addr)}")
            else:
                target = call_imms[0]
                if target == addr + size:
                    reasons.append(f"gates: call +5 PIC sequence at {hex(addr)}")
                    esp_adjust += 4
                info = _import_info(imports, target)
                if info is not None and _flags_has_pure(info[1]) and _kind_is_native_typed(info[2]):
                    if info[0]:
                        pure_callees.append(info[0])
                else:
                    reasons.append(f"gates: call at {hex(addr)} -> {hex(target)} is not a pure NATIVE_TYPED import")
                _mark_written(written, "eax")
                edx_live_out = False

        # ---- Gate 6: jumps ----
        if mn == "jmp" or mn.startswith("j") or mn.startswith("loop"):
            _check_jump_targets(addr, mn, ops, func_va, func_end, resolved, reasons)

        # ---- Gate 7: memory accesses must be stack-relative; reject escaping lea ----
        if mn == "lea":
            if len(ops) >= 2 and getattr(ops[1], "type", None) == OP_MEM:
                sbase, _sindex, _sdisp, _ = _op_mem_fields(insn, ops[1])
                dest = _op_reg_name(insn, ops[0])
                if sbase in escaped_bases:
                    reasons.append(f"gates: escaped stack address used in lea at {hex(addr)}")
                if sbase in STACK_REGS and dest not in STACK_REGS:
                    reasons.append(f"gates: escaping lea of stack slot into {dest or '?'} at {hex(addr)}")
                    if dest:
                        escaped_bases.add(dest)
        else:
            for op in ops:
                if getattr(op, "type", None) != OP_MEM:
                    continue
                base, _index, _disp, _ = _op_mem_fields(insn, op)
                if base in escaped_bases:
                    reasons.append(f"gates: escaped stack address dereferenced at {hex(addr)}")
                elif base not in STACK_REGS:
                    reasons.append(f"gates: non-stack memory access at {hex(addr)}: base={base or 'none'}")

        # ---- Gate 8: Register-entry IPA-RA (GPRs, XMM, x87) ----
        self_zero = _self_zero_reg(insn, mn, ops)
        if self_zero is not None:
            _mark_written(written, self_zero)
            if _GPR_PARENT.get(self_zero) == "edx":
                edx_live_out = False
            prev_was_push_ebp = False
            continue

        is_callee_save_push = (
            mn == "push"
            and len(ops) == 1
            and _GPR_PARENT.get(_op_reg_name(insn, ops[0]), "") in CALLEE_SAVED_REGS
        )

        # Gather registers read by this instruction.
        read_regs: List[str] = []
        if not is_callee_save_push:
            for op in ops:
                otype = getattr(op, "type", None)
                if otype == OP_REG and _is_read_op(op):
                    rname = _op_reg_name(insn, op)
                    if rname:
                        read_regs.append(rname)
                elif otype == OP_MEM:
                    base, index, _disp, _ = _op_mem_fields(insn, op)
                    if base:
                        read_regs.append(base)
                    if index:
                        read_regs.append(index)
            try:
                rr_ids, _rw_ids = insn.regs_access()
                for rid in rr_ids:
                    rname = _reg_name(insn, rid)
                    if rname:
                        read_regs.append(rname)
            except Exception:
                pass

        for rname in read_regs:
            if rname in _IGNORED_SYS_REGS:
                continue
            parent = _GPR_PARENT.get(rname)
            if parent == "ebp":
                if not has_frame_ptr and not _is_reg_written(written, rname):
                    reasons.append("regparm-entry-read:ebp")
                continue
            if parent in TRACKED_GPRS:
                if not _is_reg_written(written, rname):
                    tag = parent if parent in ENTRY_REGS else parent
                    reasons.append(f"regparm-entry-read:{tag}")
                if parent == "edx":
                    edx_live_out = False
            elif rname in _XMM_REGS:
                if not _is_reg_written(written, rname):
                    reasons.append(f"regparm-entry-read:{rname}")
            else:
                st_idx = _parse_st_index(rname)
                if st_idx is not None and st_idx >= x87_depth:
                    reasons.append(f"regparm-entry-read:st{st_idx}")

        # x87 stack depth & entry read check
        if mn in _X87_POP2_MNEMONICS:
            if x87_depth < 2:
                reasons.append("regparm-entry-read:st0")
            x87_depth = max(0, x87_depth - 2)
        elif mn in _X87_POP1_MNEMONICS:
            needed = 2 if mn in ("faddp", "fsubp", "fsubrp", "fmulp", "fdivp", "fdivrp") else 1
            if x87_depth < needed:
                reasons.append("regparm-entry-read:st0")
            x87_depth = max(0, x87_depth - 1)
        elif mn in _X87_READ_MNEMONICS:
            if x87_depth < 1:
                reasons.append("regparm-entry-read:st0")
        elif mn == "fxch":
            if x87_depth < 2:
                reasons.append("regparm-entry-read:st1")
        elif mn in ("finit", "fninit"):
            x87_depth = 0

        if mn in _X87_PUSH_MNEMONICS:
            x87_depth += 1
            if ops and getattr(ops[0], "type", None) == OP_MEM:
                sz = int(getattr(ops[0], "size", 4) or 4)
                x87_last_size = 8 if sz == 8 else 4

        # Record register writes (excluding pure callee-save pushes).
        written_now: Set[str] = set()
        for op in ops:
            if getattr(op, "type", None) == OP_REG and _is_write_op(op):
                rname = _op_reg_name(insn, op)
                if rname:
                    written_now.add(rname)
        try:
            _rr_ids, rw_ids = insn.regs_access()
            for rid in rw_ids:
                rname = _reg_name(insn, rid)
                if rname:
                    written_now.add(rname)
        except Exception:
            pass

        for rname in written_now:
            if rname in _IGNORED_SYS_REGS:
                continue
            _mark_written(written, rname)
            if _GPR_PARENT.get(rname) == "edx":
                edx_live_out = mn not in ("div", "idiv")

        # ---- Gate 9 & 10: Stack frame tracking, ret imm16, and argument slots ----
        if mn == "mov" and len(ops) >= 2:
            if _op_reg_name(insn, ops[0]) == "ebp" and _op_reg_name(insn, ops[1]) == "esp" and prev_was_push_ebp:
                has_frame_ptr = True
                _mark_written(written, "ebp")

        prev_was_push_ebp = (mn == "push" and len(ops) == 1 and _op_reg_name(insn, ops[0]) == "ebp")

        # Parameter slot reads BEFORE adjusting esp for this instruction
        for op in ops:
            if getattr(op, "type", None) != OP_MEM:
                continue
            base, _index, disp, _ = _op_mem_fields(insn, op)
            op_size = int(getattr(op, "size", 4) or 4)
            if base == "ebp" and has_frame_ptr and disp >= 8:
                slot = (disp - 8) // 4
                arg_slots.setdefault(slot, _infer_arg_slot_type(mn, op_size))
            elif base == "esp" and disp >= 4 + esp_adjust:
                slot = (disp - 4 - esp_adjust) // 4
                arg_slots.setdefault(slot, _infer_arg_slot_type(mn, op_size))

        if mn == "push":
            esp_adjust += 4
        elif mn == "pop":
            esp_adjust = max(0, esp_adjust - 4)
        elif mn == "sub" and len(ops) >= 2 and _op_reg_name(insn, ops[0]) == "esp":
            imm = _op_imm(ops[1])
            if imm is not None and imm > 0:
                esp_adjust += imm
                local_frame += imm
                max_frame = max(max_frame, local_frame)
        elif mn == "add" and len(ops) >= 2 and _op_reg_name(insn, ops[0]) == "esp":
            imm = _op_imm(ops[1])
            if imm is not None and imm > 0:
                esp_adjust = max(0, esp_adjust - imm)
                local_frame = max(0, local_frame - imm)
        elif mn == "enter":
            imm = _op_imm(ops[0]) if ops else None
            if imm is not None and imm > 0:
                max_frame = max(max_frame, imm)
        elif mn == "leave":
            local_frame = 0
            esp_adjust = 0
        elif mn == "ret":
            saw_ret = True
            if any(_op_imm(op) is not None for op in ops):
                reasons.append(f"gates: ret imm16 at {hex(addr)} (stdcall/sret callee-pop)")
            if "eax" in escaped_bases:
                reasons.append(f"gates: stack address escaped via return at {hex(addr)}")

    if max_frame > max_stack_frame:
        reasons.append(f"gates: stack frame {max_frame} exceeds max_stack_frame={max_stack_frame}")

    # Reconstruct ordered cdecl parameter list (handling 64-bit 2-slot args & gaps)
    arg_list: List[str] = []
    if arg_slots:
        max_slot = max(arg_slots)
        slot_idx = 0
        while slot_idx <= max_slot:
            stype = arg_slots.get(slot_idx, ScalarType.I32.value)
            arg_list.append(stype)
            slot_idx += 2 if stype in (ScalarType.F64.value, ScalarType.I64.value, ScalarType.U64.value) else 1

    if len(arg_list) > MAX_SCALAR_PARAMS:
        reasons.append(f"gates: {len(arg_list)} scalar parameters exceeds limit {MAX_SCALAR_PARAMS}")
        arg_list = arg_list[:MAX_SCALAR_PARAMS]

    if saw_ret:
        if x87_depth > 0:
            ret_type = ScalarType.F64.value if x87_last_size == 8 else ScalarType.F32.value
        elif any(r in written for r in ("eax", "ax", "al")):
            ret_type = ScalarType.I64.value if edx_live_out else ScalarType.I32.value
        else:
            ret_type = ScalarType.VOID.value
    else:
        ret_type = ScalarType.VOID.value

    deduped = list(dict.fromkeys(reasons))
    return GateVerdict(
        passed=not deduped,
        reasons=deduped,
        ret_type=ret_type,
        arg_types=arg_list,
        stack_frame_bytes=max_frame,
        pure_callees=sorted(set(pure_callees)),
    )


def _run_self_tests() -> bool:
    print("[gates] Running inline self-tests...")
    v_empty = evaluate_purity_gate("empty", 0x1000, b"")
    assert not v_empty.passed
    if capstone is None:
        # Fail-closed path is the intended behavior without capstone; skip the
        # disassembly-based checks since they cannot run here.
        print("[gates] capstone unavailable — skipping disassembly checks (fail-closed verified)")
        print("[gates] Inline self-tests passed!")
        return True
    # plugin_compute machine bytes: mov eax,[esp+8]; imul eax,[esp+4]; add eax,0xc; ret
    v_pure = evaluate_purity_gate(
        "plugin_compute",
        0x1130,
        bytes.fromhex("8b4424080faf44240483c00cc3"),
        seg_ranges=[(0x1000, 0x2000)],
        known_imports={},
    )
    assert v_pure.passed, f"expected plugin_compute pass, got {v_pure.reasons}"
    assert v_pure.ret_type == "i32" and v_pure.arg_types == ["i32", "i32"], v_pure
    # regparm read: mov eax, edx; ret
    v_reg = evaluate_purity_gate("regparm", 0x1000, bytes.fromhex("89d0c3"), seg_ranges=[], known_imports={})
    assert not v_reg.passed and "regparm-entry-read:edx" in v_reg.reasons, v_reg.reasons
    # self-zero xor eax, eax; ret
    v_zero = evaluate_purity_gate("zero", 0x1000, bytes.fromhex("31c0c3"), seg_ranges=[], known_imports={})
    assert v_zero.passed, v_zero.reasons
    print("[gates] Inline self-tests passed!")
    return True


if __name__ == "__main__":
    _run_self_tests()
