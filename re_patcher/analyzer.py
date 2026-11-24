"""Stage 1 — Static Reverse-Engineering Analysis (elftools, capstone, angr/cle, r2, ghidra)."""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
import os
import shutil
import struct
import subprocess
import tempfile
from typing import Dict, List, Optional, Set, Tuple

import capstone
from capstone import x86 as cs_x86
from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection
from elftools.elf.sections import SymbolTableSection

from . import PatcherError

# Forbidden Long-Mode / privileged / BCD opcodes (N07 & N08)
PRIVILEGED_INSN_IDS = {
    cs_x86.X86_INS_HLT,
    cs_x86.X86_INS_UD2,
    cs_x86.X86_INS_UD0,
    cs_x86.X86_INS_UD1,
    cs_x86.X86_INS_IN,
    cs_x86.X86_INS_OUT,
    cs_x86.X86_INS_INSB,
    cs_x86.X86_INS_INSW,
    cs_x86.X86_INS_INSD,
    cs_x86.X86_INS_OUTSB,
    cs_x86.X86_INS_OUTSW,
    cs_x86.X86_INS_OUTSD,
    cs_x86.X86_INS_CLI,
    cs_x86.X86_INS_STI,
    cs_x86.X86_INS_LIDT,
    cs_x86.X86_INS_LGDT,
    cs_x86.X86_INS_LLDT,
    cs_x86.X86_INS_LTR,
    cs_x86.X86_INS_SIDT,
    cs_x86.X86_INS_SGDT,
    cs_x86.X86_INS_SLDT,
    cs_x86.X86_INS_STR,
    cs_x86.X86_INS_LMSW,
    cs_x86.X86_INS_SMSW,
    cs_x86.X86_INS_CLTS,
    cs_x86.X86_INS_INVD,
    cs_x86.X86_INS_WBINVD,
    cs_x86.X86_INS_INVLPG,
    cs_x86.X86_INS_RDMSR,
    cs_x86.X86_INS_WRMSR,
    cs_x86.X86_INS_RDPMC,
    cs_x86.X86_INS_SYSRET,
    cs_x86.X86_INS_SYSEXIT,
    cs_x86.X86_INS_SWAPGS,
    cs_x86.X86_INS_IRET,
    cs_x86.X86_INS_IRETD,
    cs_x86.X86_INS_IRETQ,
    cs_x86.X86_INS_INT1,
    cs_x86.X86_INS_INTO,
}

BCD_INVALID_LONG_MODE_IDS = {
    cs_x86.X86_INS_AAA,
    cs_x86.X86_INS_AAS,
    cs_x86.X86_INS_DAA,
    cs_x86.X86_INS_DAS,
    cs_x86.X86_INS_AAM,
    cs_x86.X86_INS_AAD,
    cs_x86.X86_INS_SALC,
    cs_x86.X86_INS_ARPL,
    cs_x86.X86_INS_BOUND,
    cs_x86.X86_INS_LES,
    cs_x86.X86_INS_LDS,
}

# Raw single-byte segment push/pop invalid in Long Mode
INVALID_LONG_MODE_RAW_BYTES = {0x06, 0x07, 0x0E, 0x16, 0x17, 0x1E, 0x1F}

NORETURN_SYMS = {
    "__libc_start_main",
    "exit",
    "_exit",
    "abort",
    "__stack_chk_fail",
    "__stack_chk_fail_local",
    "pthread_exit",
}


@dataclass
class PhdrInfo:
    p_type: int
    p_offset: int
    p_vaddr: int
    p_paddr: int
    p_filesz: int
    p_memsz: int
    p_flags: int
    p_align: int


@dataclass
class ShdrInfo:
    index: int
    name: str
    sh_type: int
    sh_flags: int
    sh_addr: int
    sh_offset: int
    sh_size: int
    sh_link: int
    sh_info: int
    sh_addralign: int
    sh_entsize: int


@dataclass
class DynSymInfo:
    index: int
    name: str
    st_value: int
    st_size: int
    st_info: int
    st_other: int
    st_shndx: int

    @property
    def bind(self) -> int:
        return (self.st_info >> 4) & 0xF

    @property
    def sym_type(self) -> int:
        return self.st_info & 0xF


@dataclass
class RelocInfo:
    offset: int
    sym_idx: int
    r_type: int
    sym_name: str


@dataclass
class PltStubInfo:
    va: int
    size: int
    sym_name: str
    got_slot_va: int


@dataclass
class JumpTableInfo:
    jmp_va: int
    replace_start_va: int
    replace_end_va: int
    tbl_va: int
    is_pic: bool
    got_reg: Optional[int]  # capstone X86_REG_*
    idx_reg: int            # capstone X86_REG_*
    targets: List[int]


@dataclass
class CodeRange:
    name: str
    va: int
    offset: int
    data: bytes


@dataclass
class AnalysisResult:
    path: str
    raw_bytes: bytes
    e_type: int
    e_entry: int
    has_interp: bool
    interp_str: str
    has_symtab: bool
    phdrs: List[PhdrInfo]
    shdrs: List[ShdrInfo]
    dyn_tags: Dict[int, int]
    dt_needed: List[str]
    dynsyms: List[DynSymInfo]
    rel_dyn: List[RelocInfo]
    rel_plt: List[RelocInfo]
    got_va: int
    got_plt_va: int
    plt_stubs: Dict[int, PltStubInfo]
    code_ranges: List[CodeRange]
    init_va: int
    fini_va: int
    init_array: List[int]
    fini_array: List[int]
    tls_info: Optional[Tuple[int, int, int, int, int]]  # vaddr, offset, filesz, memsz, align
    functions: Set[int]
    sym_names: Dict[int, str]
    pic_thunks: Dict[int, str]  # thunk_va -> reg name ('ebx', 'ecx', ...)
    jump_tables: Dict[int, JumpTableInfo]  # jmp_va -> JumpTableInfo
    syscalls: Dict[int, str]  # insn_va -> "int80"|"sysenter"
    addr_taken_funcs: Set[int]
    reachable_insns: Dict[int, capstone.CsInsn]
    func_classifications: Dict[int, str]  # func_va -> REWRITABLE_NATIVE | REQUIRES_DBT_FALLBACK
    max_vaddr: int
    symbol_versions: List[Tuple[str, str, str]] = field(default_factory=list)  # (soname, sym_name, version)
    sym_version_map: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # sym_name -> (soname, version)


CALLEE_SAVED_REGS = {
    cs_x86.X86_REG_EBX,
    cs_x86.X86_REG_ESI,
    cs_x86.X86_REG_EDI,
    cs_x86.X86_REG_EBP,
}
CALLER_SAVED_REGS = {
    cs_x86.X86_REG_EAX,
    cs_x86.X86_REG_ECX,
    cs_x86.X86_REG_EDX,
}


class GotProvenanceTracker:
    """Tracks which register(s) currently hold the GOT base VA (got_va)."""

    def __init__(self, got_va: int, pic_thunks: Dict[int, str], functions: Set[int]):
        self.got_va = got_va
        self.pic_thunks = pic_thunks
        self.functions = functions
        self.got_regs: Set[int] = set()
        self.func_got_reg: Optional[int] = None
        self.pending_pc: Dict[int, int] = {}  # reg_id -> pc_va

    def is_got_reg(self, reg_id: int) -> bool:
        if self.got_va == 0:
            return False
        return (reg_id in self.got_regs) or (self.func_got_reg is not None and reg_id == self.func_got_reg)

    def on_instruction(
        self, insn: capstone.CsInsn, next_insn: Optional[capstone.CsInsn] = None
    ) -> None:
        if self.got_va == 0:
            return

        # Start of genuine function boundary clears provenance (never clear on `call $+5; pop %reg` landing pad)
        if insn.address in self.functions and insn.address not in self.pending_pc.values():
            self.got_regs.clear()
            self.pending_pc.clear()
            self.func_got_reg = None

        # Call __x86.get_pc_thunk.<reg> or inline call $+5
        if insn.id == cs_x86.X86_INS_CALL:
            if len(insn.operands) == 1 and insn.operands[0].type == cs_x86.X86_OP_IMM:
                tgt = insn.operands[0].imm & 0xFFFFFFFF
                if tgt in self.pic_thunks:
                    rname = self.pic_thunks[tgt]
                    cs_reg = {
                        "eax": cs_x86.X86_REG_EAX,
                        "ecx": cs_x86.X86_REG_ECX,
                        "edx": cs_x86.X86_REG_EDX,
                        "ebx": cs_x86.X86_REG_EBX,
                        "esp": cs_x86.X86_REG_ESP,
                        "ebp": cs_x86.X86_REG_EBP,
                        "esi": cs_x86.X86_REG_ESI,
                        "edi": cs_x86.X86_REG_EDI,
                    }.get(rname)
                    if cs_reg is not None:
                        self.pending_pc[cs_reg] = (insn.address + insn.size) & 0xFFFFFFFF
                    return
                elif (insn.operands[0].imm & 0xFFFFFFFF) == (insn.address + insn.size):
                    # call $+5; next insn is pop %reg
                    if (
                        next_insn is not None
                        and next_insn.id == cs_x86.X86_INS_POP
                        and len(next_insn.operands) == 1
                        and next_insn.operands[0].type == cs_x86.X86_OP_REG
                    ):
                        self.pending_pc[next_insn.operands[0].reg] = (
                            insn.address + insn.size
                        ) & 0xFFFFFFFF
                    return
            # Caller-saved registers are clobbered by calls — NEVER invalidate callee-saved regs!
            for r in CALLER_SAVED_REGS:
                self.got_regs.discard(r)
                self.pending_pc.pop(r, None)
            return

        # Pop into register (if not handled by call $+5)
        if insn.id == cs_x86.X86_INS_POP and len(insn.operands) == 1:
            op = insn.operands[0]
            if op.type == cs_x86.X86_OP_REG:
                if op.reg not in self.pending_pc:
                    self.got_regs.discard(op.reg)
                    if op.reg == self.func_got_reg:
                        self.func_got_reg = None
            return

        # add $imm, %reg
        if insn.id == cs_x86.X86_INS_ADD and len(insn.operands) == 2:
            dst, src = insn.operands[0], insn.operands[1]
            if dst.type == cs_x86.X86_OP_REG:
                if src.type == cs_x86.X86_OP_IMM:
                    imm = src.imm & 0xFFFFFFFF
                    if dst.reg in self.pending_pc:
                        pc_val = self.pending_pc.pop(dst.reg)
                        if ((pc_val + imm) & 0xFFFFFFFF) == self.got_va:
                            self.got_regs.add(dst.reg)
                            if dst.reg in CALLEE_SAVED_REGS:
                                self.func_got_reg = dst.reg
                    elif dst.reg in self.got_regs:
                        if imm != 0:
                            self.got_regs.discard(dst.reg)
                            if dst.reg == self.func_got_reg:
                                self.func_got_reg = None
                else:
                    self.got_regs.discard(dst.reg)
                    self.pending_pc.pop(dst.reg, None)
                    if dst.reg == self.func_got_reg:
                        self.func_got_reg = None
            return

        # mov %got_reg, %reg (register copy)
        if insn.id == cs_x86.X86_INS_MOV and len(insn.operands) == 2:
            dst, src = insn.operands[0], insn.operands[1]
            if dst.type == cs_x86.X86_OP_REG:
                if src.type == cs_x86.X86_OP_REG and self.is_got_reg(src.reg):
                    self.got_regs.add(dst.reg)
                else:
                    self.got_regs.discard(dst.reg)
                    if dst.reg == self.func_got_reg:
                        self.func_got_reg = None
                self.pending_pc.pop(dst.reg, None)
            return

        # On jmp / jcc / ret: NEVER invalidate callee-saved GOT registers!
        if insn.id in (
            cs_x86.X86_INS_JMP,
            cs_x86.X86_INS_RET,
            cs_x86.X86_INS_RETF,
        ) or (insn.mnemonic.startswith("j") and insn.mnemonic != "jmp"):
            for r in CALLER_SAVED_REGS:
                self.got_regs.discard(r)
                self.pending_pc.pop(r, None)
            if insn.id in (cs_x86.X86_INS_RET, cs_x86.X86_INS_RETF):
                self.func_got_reg = None
            return

        # Any other instruction writing to operand 0 register invalidates it
        if insn.id not in (cs_x86.X86_INS_CMP, cs_x86.X86_INS_TEST, cs_x86.X86_INS_PUSH):
            if len(insn.operands) > 0 and insn.operands[0].type == cs_x86.X86_OP_REG:
                self.got_regs.discard(insn.operands[0].reg)
                self.pending_pc.pop(insn.operands[0].reg, None)
                if insn.operands[0].reg == self.func_got_reg:
                    self.func_got_reg = None


def _vaddr_to_bytes(phdrs: List[PhdrInfo], data: bytes, va: int, size: int) -> Optional[bytes]:
    for ph in phdrs:
        if ph.p_type == 1 and ph.p_filesz > 0:
            if ph.p_vaddr <= va and va + size <= ph.p_vaddr + ph.p_filesz:
                off = ph.p_offset + (va - ph.p_vaddr)
                return data[off : off + size]
    return None


def _vaddr_to_file_offset(phdrs: List[PhdrInfo], va: int) -> Optional[int]:
    for ph in phdrs:
        if ph.p_type == 1 and ph.p_filesz > 0:
            if ph.p_vaddr <= va < ph.p_vaddr + ph.p_filesz:
                return ph.p_offset + (va - ph.p_vaddr)
    return None


def _read_u32(phdrs: List[PhdrInfo], data: bytes, va: int) -> Optional[int]:
    b = _vaddr_to_bytes(phdrs, data, va, 4)
    if b is None or len(b) < 4:
        return None
    return struct.unpack_from("<I", b, 0)[0]


def _read_i32(phdrs: List[PhdrInfo], data: bytes, va: int) -> Optional[int]:
    b = _vaddr_to_bytes(phdrs, data, va, 4)
    if b is None or len(b) < 4:
        return None
    return struct.unpack_from("<i", b, 0)[0]


def _cs_reg_to_32_name(insn: capstone.CsInsn, reg_id: int) -> Optional[str]:
    name = insn.reg_name(reg_id)
    if name in ("eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"):
        return name
    return None


def _is_forbidden_insn(insn: capstone.CsInsn) -> Optional[str]:
    if insn.id in PRIVILEGED_INSN_IDS:
        return f"privileged/trap instruction '{insn.mnemonic} {insn.op_str}'"
    if insn.id in BCD_INVALID_LONG_MODE_IDS:
        return f"Long-Mode invalid instruction '{insn.mnemonic} {insn.op_str}'"
    if insn.id == cs_x86.X86_INS_INT:
        if len(insn.bytes) >= 2 and insn.bytes[0] == 0xCD:
            imm = insn.bytes[1]
            if imm not in (0x80, 0x03):
                return f"unsupported software interrupt 'int 0x{imm:02x}'"
        elif insn.bytes[0] == 0xCE:
            return "Long-Mode invalid instruction 'into'"
    # Check segment register push/pop (0x06, 0x07, 0x0E, 0x16, 0x17, 0x1E, 0x1F)
    if len(insn.bytes) == 1 and insn.bytes[0] in INVALID_LONG_MODE_RAW_BYTES:
        return f"Long-Mode invalid segment stack instruction '{insn.mnemonic} {insn.op_str}'"
    return None


def _run_r2_analysis(path: str, verbose: bool = False) -> Tuple[Set[int], Dict[int, List[int]]]:
    """Run radare2 / rizin headless analysis (`aaa; aflj`) for stripped binaries / CFG corroboration."""
    r2_bin = (
        shutil.which("rizin")
        or ("/usr/bin/rizin" if os.path.exists("/usr/bin/rizin") else None)
        or shutil.which("r2")
        or shutil.which("radare2")
    )
    if not r2_bin:
        return set(), {}
    funcs: Set[int] = set()
    switches: Dict[int, List[int]] = {}
    try:
        proc = subprocess.run(
            [r2_bin, "-e", "scr.color=0", "-q", "-c", "aaa; aflj", path],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            text = proc.stdout.strip()
            start = text.find("[")
            if start >= 0:
                arr = json.loads(text[start:])
                for item in arr:
                    if isinstance(item, dict) and "offset" in item:
                        va = int(item["offset"])
                        if va > 0:
                            funcs.add(va)
    except Exception as exc:
        if verbose:
            import sys
            sys.stderr.write(f"[newminhh:patcher] WARN: r2 analysis skipped: {exc}\n")
    return funcs, switches


def _run_ghidra_analysis(path: str, verbose: bool = False) -> Tuple[Set[int], Dict[int, List[int]]]:
    """Run Ghidra headless analyzer when --use-ghidra is requested."""
    headless = "/opt/ghidra/support/analyzeHeadless"
    if not os.path.exists(headless):
        raise PatcherError(f"--use-ghidra requested but '{headless}' was not found")

    funcs: Set[int] = set()
    switches: Dict[int, List[int]] = {}
    with tempfile.TemporaryDirectory(prefix="nm_ghidra_") as tmpdir:
        proj_dir = os.path.join(tmpdir, "proj")
        os.makedirs(proj_dir, exist_ok=True)
        out_json = os.path.join(tmpdir, "ghidra_out.json")
        script_path = os.path.join(tmpdir, "DumpCfg.py")
        with open(script_path, "w", encoding="utf-8") as sf:
            sf.write(
                """import json
fm = currentProgram.getFunctionManager()
funcs = []
for f in fm.getFunctions(True):
    funcs.append(int(f.getEntryPoint().getOffset()))
with open(%r, "w") as out:
    json.dump({"functions": funcs}, out)
"""
                % out_json
            )
        proc = subprocess.run(
            [
                headless,
                proj_dir,
                "nmproj",
                "-import",
                path,
                "-postScript",
                script_path,
                "-deleteProject",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if os.path.exists(out_json):
            with open(out_json, "r", encoding="utf-8") as jf:
                obj = json.load(jf)
                for va in obj.get("functions", []):
                    funcs.add(int(va))
        elif verbose:
            import sys
            sys.stderr.write(
                f"[newminhh:patcher] WARN: ghidra produced no JSON (rc={proc.returncode})\n"
            )
    return funcs, switches


def _run_angr_cfg(path: str, verbose: bool = False):
    """Run angr CFGFast with cle auto_load_libs=False."""
    try:
        import logging
        logging.getLogger("angr").setLevel(logging.ERROR)
        logging.getLogger("cle").setLevel(logging.ERROR)
        logging.getLogger("pyvex").setLevel(logging.ERROR)
        import angr

        proj = angr.Project(path, auto_load_libs=False, load_options={"main_opts": {"base_addr": 0}})
        # Use project's natural base_addr for ET_EXEC vs ET_DYN
        # Actually cle loads ET_DYN at 0x400000 by default unless main_opts={'base_addr': 0}
        # Setting base_addr=0 for ET_DYN aligns angr VAs 1:1 with ELF file VAs!
        with open(path, "rb") as f:
            e_type = struct.unpack_from("<H", f.read(20), 16)[0]
        if e_type == 2:  # ET_EXEC
            proj = angr.Project(path, auto_load_libs=False)
        cfg = proj.analyses.CFGFast(normalize=True, resolve_indirect_jumps=True)
        return proj, cfg
    except Exception as exc:
        if verbose:
            import sys
            sys.stderr.write(f"[newminhh:patcher] WARN: angr CFGFast fallback: {exc}\n")
        return None, None


def analyze_binary(
    path: str,
    raw_bytes: bytes,
    use_r2: bool = False,
    use_ghidra: bool = False,
    verbose: bool = False,
) -> AnalysisResult:
    elf = ELFFile(io.BytesIO(raw_bytes))
    e_type = struct.unpack_from("<H", raw_bytes, 16)[0]
    e_entry = elf.header["e_entry"]

    phdrs: List[PhdrInfo] = []
    has_interp = False
    interp_str = ""
    tls_info: Optional[Tuple[int, int, int, int, int]] = None
    max_vaddr = 0
    for seg in elf.iter_segments():
        h = seg.header
        ph = PhdrInfo(
            p_type=seg["p_type"] if isinstance(seg["p_type"], int) else h.p_type,
            p_offset=h.p_offset,
            p_vaddr=h.p_vaddr,
            p_paddr=h.p_paddr,
            p_filesz=h.p_filesz,
            p_memsz=h.p_memsz,
            p_flags=h.p_flags,
            p_align=h.p_align,
        )
        # Normalise p_type to integer using raw phdr table
        idx = len(phdrs)
        e_phoff = elf.header["e_phoff"]
        e_phentsize = elf.header["e_phentsize"]
        raw_ptype = struct.unpack_from("<I", raw_bytes, e_phoff + idx * e_phentsize)[0]
        ph.p_type = raw_ptype
        phdrs.append(ph)
        if raw_ptype == 1:  # PT_LOAD
            max_vaddr = max(max_vaddr, ph.p_vaddr + ph.p_memsz)
        elif raw_ptype == 3:  # PT_INTERP
            has_interp = True
            interp_str = raw_bytes[ph.p_offset : ph.p_offset + ph.p_filesz].rstrip(b"\x00").decode(
                "ascii", errors="replace"
            )
        elif raw_ptype == 7:  # PT_TLS
            tls_info = (ph.p_vaddr, ph.p_offset, ph.p_filesz, ph.p_memsz, ph.p_align)

    shdrs: List[ShdrInfo] = []
    has_symtab = False
    sym_names: Dict[int, str] = {}
    code_ranges: List[CodeRange] = []
    dyn_tags: Dict[int, int] = {}
    dt_needed: List[str] = []
    dynsyms: List[DynSymInfo] = []
    rel_dyn: List[RelocInfo] = []
    rel_plt: List[RelocInfo] = []

    for idx, sec in enumerate(elf.iter_sections()):
        sh = sec.header
        raw_shtype = struct.unpack_from("<I", raw_bytes, elf.header["e_shoff"] + idx * elf.header["e_shentsize"] + 4)[0] if elf.header["e_shoff"] else 0
        shdrs.append(
            ShdrInfo(
                index=idx,
                name=sec.name,
                sh_type=raw_shtype,
                sh_flags=sh["sh_flags"],
                sh_addr=sh["sh_addr"],
                sh_offset=sh["sh_offset"],
                sh_size=sh["sh_size"],
                sh_link=sh["sh_link"],
                sh_info=sh["sh_info"],
                sh_addralign=sh["sh_addralign"],
                sh_entsize=sh["sh_entsize"],
            )
        )
        if sec.name == ".symtab" and isinstance(sec, SymbolTableSection):
            has_symtab = True
            for sym in sec.iter_symbols():
                if sym.name and sym["st_value"] != 0 and sym["st_info"]["type"] == "STT_FUNC":
                    sym_names[sym["st_value"]] = sym.name
        elif sec.name in (".init", ".text", ".fini") and sh["sh_size"] > 0:
            code_ranges.append(
                CodeRange(name=sec.name, va=sh["sh_addr"], offset=sh["sh_offset"], data=sec.data())
            )

    # Parse PT_DYNAMIC directly from raw_bytes
    dyn_ph = next((p for p in phdrs if p.p_type == 2), None)
    dynstr_bytes = b""
    if dyn_ph is not None:
        for pos in range(dyn_ph.p_offset, dyn_ph.p_offset + dyn_ph.p_filesz - 7, 8):
            d_tag, d_val = struct.unpack_from("<iI", raw_bytes, pos)
            if d_tag == 0:
                break
            if d_tag == 1:  # DT_NEEDED
                dt_needed.append(str(d_val))
            else:
                dyn_tags[d_tag] = d_val

        if 5 in dyn_tags and 10 in dyn_tags:
            b = _vaddr_to_bytes(phdrs, raw_bytes, dyn_tags[5], dyn_tags[10])
            if b is not None:
                dynstr_bytes = b

    if not dynstr_bytes:
        dynstr_sec = elf.get_section_by_name(".dynstr")
        if dynstr_sec is not None:
            dynstr_bytes = dynstr_sec.data()

    def _get_dynstr(off: int) -> str:
        if off >= len(dynstr_bytes):
            return ""
        end = dynstr_bytes.find(b"\x00", off)
        if end < 0:
            end = len(dynstr_bytes)
        return dynstr_bytes[off:end].decode("ascii", errors="replace")

    if dyn_ph is not None:
        dt_needed = [_get_dynstr(int(x)) for x in dt_needed]

        # Parse .dynsym
        dynsym_sec = elf.get_section_by_name(".dynsym")
        if dynsym_sec is not None and isinstance(dynsym_sec, SymbolTableSection):
            for s_idx, sym in enumerate(dynsym_sec.iter_symbols()):
                st_info = (
                    ({"STB_LOCAL": 0, "STB_GLOBAL": 1, "STB_WEAK": 2}.get(sym["st_info"]["bind"], 1) << 4)
                    | {"STT_NOTYPE": 0, "STT_OBJECT": 1, "STT_FUNC": 2, "STT_SECTION": 3, "STT_FILE": 4, "STT_TLS": 6, "STT_GNU_IFUNC": 10}.get(sym["st_info"]["type"], 0)
                )
                shndx_raw = sym["st_shndx"]
                if isinstance(shndx_raw, str):
                    shndx = {"SHN_UNDEF": 0, "SHN_ABS": 0xFFF1, "SHN_COMMON": 0xFFF2}.get(shndx_raw, 0)
                else:
                    shndx = int(shndx_raw)
                dynsyms.append(
                    DynSymInfo(
                        index=s_idx,
                        name=sym.name,
                        st_value=sym["st_value"],
                        st_size=sym["st_size"],
                        st_info=st_info,
                        st_other=sym["st_other"]["visibility"] if isinstance(sym["st_other"]["visibility"], int) else 0,
                        st_shndx=shndx,
                    )
                )
                if sym.name and sym["st_value"] != 0 and shndx != 0 and (st_info & 0xF) == 2:
                    sym_names.setdefault(sym["st_value"], sym.name)

        # Parse .rel.dyn / .rel.plt via dynamic tags (works even on stripped sections!)
        def _parse_rel_table(va: int, sz: int, ent: int) -> List[RelocInfo]:
            out: List[RelocInfo] = []
            tbl = _vaddr_to_bytes(phdrs, raw_bytes, va, sz)
            if tbl is None:
                return out
            for rpos in range(0, len(tbl) - ent + 1, ent):
                r_off, r_info = struct.unpack_from("<II", tbl, rpos)
                s_idx = r_info >> 8
                r_type = r_info & 0xFF
                sname = dynsyms[s_idx].name if s_idx < len(dynsyms) else ""
                out.append(RelocInfo(offset=r_off, sym_idx=s_idx, r_type=r_type, sym_name=sname))
            return out

        if 17 in dyn_tags and 18 in dyn_tags:
            rel_dyn = _parse_rel_table(dyn_tags[17], dyn_tags[18], dyn_tags.get(19, 8))
        if 23 in dyn_tags and 2 in dyn_tags:
            rel_plt = _parse_rel_table(dyn_tags[23], dyn_tags[2], 8)

    # Parse ELF symbol versioning (.gnu.versym, .gnu.verneed / SHT_GNU_verneed)
    versym_sec = elf.get_section_by_name(".gnu.versym")
    verneed_sec = elf.get_section_by_name(".gnu.verneed")
    versym_data = versym_sec.data() if versym_sec is not None else b""
    verneed_data = verneed_sec.data() if verneed_sec is not None else b""

    if not verneed_data and 0x6FFFFFFE in dyn_tags:  # DT_VERNEED
        vn_va = dyn_tags[0x6FFFFFFE]
        b = _vaddr_to_bytes(phdrs, raw_bytes, vn_va, 4096)
        if b:
            verneed_data = b
    if not versym_data and 0x6FFFFFF0 in dyn_tags:  # DT_VERSYM
        vs_va = dyn_tags[0x6FFFFFF0]
        vs_sz = len(dynsyms) * 2
        b = _vaddr_to_bytes(phdrs, raw_bytes, vs_va, vs_sz)
        if b:
            versym_data = b

    ver_map: Dict[int, Tuple[str, str]] = {}
    if verneed_data and dynstr_bytes:
        pos = 0
        while pos + 16 <= len(verneed_data):
            vn_version, vn_cnt, vn_file, vn_aux, vn_next = struct.unpack_from("<HHIII", verneed_data, pos)
            soname = _get_dynstr(vn_file)
            aux_pos = pos + vn_aux
            for _ in range(vn_cnt):
                if aux_pos + 16 > len(verneed_data):
                    break
                vna_hash, vna_flags, vna_other, vna_name, vna_next = struct.unpack_from(
                    "<IHHI I", verneed_data, aux_pos
                )
                v_idx = vna_other & 0x7FFF
                v_name = _get_dynstr(vna_name)
                ver_map[v_idx] = (soname, v_name)
                if vna_next == 0:
                    break
                aux_pos += vna_next
            if vn_next == 0:
                break
            pos += vn_next

    symbol_versions: List[Tuple[str, str, str]] = []
    sym_version_map: Dict[str, Tuple[str, str]] = {}
    if versym_data and ver_map:
        vs_count = len(versym_data) // 2
        for s_idx in range(min(len(dynsyms), vs_count)):
            sym = dynsyms[s_idx]
            v_val = struct.unpack_from("<H", versym_data, s_idx * 2)[0]
            v_idx = v_val & 0x7FFF
            if sym.name and v_idx in ver_map:
                soname, ver_str = ver_map[v_idx]
                symbol_versions.append((soname, sym.name, ver_str))
                sym_version_map[sym.name] = (soname, ver_str)

    # Identify GOT / GOT.PLT VA
    got_plt_va = dyn_tags.get(3, 0)
    got_sec = elf.get_section_by_name(".got")
    got_plt_sec = elf.get_section_by_name(".got.plt")
    if got_plt_va == 0 and got_plt_sec is not None:
        got_plt_va = got_plt_sec["sh_addr"]
    got_va = got_plt_va if got_plt_va != 0 else (got_sec["sh_addr"] if got_sec is not None else 0)

    # Map .plt (and .plt.got) stubs to symbol names & GOT slots
    plt_stubs: Dict[int, PltStubInfo] = {}
    jmp_slot_by_got: Dict[int, RelocInfo] = {r.offset: r for r in rel_plt}
    glob_dat_by_got: Dict[int, RelocInfo] = {r.offset: r for r in rel_dyn if r.r_type == 6}

    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
    md.detail = True

    for plt_name in (".plt", ".plt.got"):
        psec = elf.get_section_by_name(plt_name)
        if psec is None or psec["sh_size"] == 0:
            continue
        p_va = psec["sh_addr"]
        p_data = psec.data()
        insns = list(md.disasm(p_data, p_va))
        for i, insn in enumerate(insns):
            if insn.id == cs_x86.X86_INS_JMP and len(insn.operands) == 1:
                op = insn.operands[0]
                if op.type == cs_x86.X86_OP_MEM:
                    mem = op.mem
                    slot_va = None
                    if mem.base == cs_x86.X86_REG_INVALID and mem.index == cs_x86.X86_REG_INVALID:
                        slot_va = mem.disp & 0xFFFFFFFF
                    elif mem.base == cs_x86.X86_REG_EBX and mem.index == cs_x86.X86_REG_INVALID:
                        slot_va = (got_va + mem.disp) & 0xFFFFFFFF
                    if slot_va is not None:
                        rel = jmp_slot_by_got.get(slot_va) or glob_dat_by_got.get(slot_va)
                        if rel is not None:
                            sname = rel.sym_name or (f"__nm_ifunc_{slot_va:x}" if rel.r_type == 42 else "")
                            if sname:
                                plt_stubs[insn.address] = PltStubInfo(
                                    va=insn.address,
                                    size=16 if plt_name == ".plt" else insn.size,
                                    sym_name=sname,
                                    got_slot_va=slot_va,
                                )

    # Extract init / fini / init_array / fini_array
    init_va = dyn_tags.get(12, 0)
    fini_va = dyn_tags.get(13, 0)
    rel_dyn_map = {r.offset: r for r in rel_dyn}

    # If DT_INIT_ARRAY (25) or DT_INIT_ARRAYSZ (27) missing, check section headers
    if 25 not in dyn_tags or 27 not in dyn_tags:
        for sh in shdrs:
            if (sh.name == ".init_array" or sh.sh_type == 14) and sh.sh_size > 0:
                dyn_tags[25] = sh.sh_addr
                dyn_tags[27] = sh.sh_size
                break

    # If DT_FINI_ARRAY (26) or DT_FINI_ARRAYSZ (28) missing, check section headers
    if 26 not in dyn_tags or 28 not in dyn_tags:
        for sh in shdrs:
            if (sh.name == ".fini_array" or sh.sh_type == 15) and sh.sh_size > 0:
                dyn_tags[26] = sh.sh_addr
                dyn_tags[28] = sh.sh_size
                break

    def _read_ptr_array(arr_tag: int, sz_tag: int) -> List[int]:
        res: List[int] = []
        if arr_tag not in dyn_tags or sz_tag not in dyn_tags:
            return res
        ava = dyn_tags[arr_tag]
        asz = dyn_tags[sz_tag]
        raw = _vaddr_to_bytes(phdrs, raw_bytes, ava, asz)
        if raw is None:
            return res
        for off in range(0, len(raw) - 3, 4):
            val = struct.unpack_from("<I", raw, off)[0]
            if val not in (0, 0xFFFFFFFF):
                res.append(val)
        return res

    init_array = _read_ptr_array(25, 27)
    fini_array = _read_ptr_array(26, 28)

    # Disassemble all code_ranges linearly first to map instruction boundaries
    all_code_insns: Dict[int, capstone.CsInsn] = {}
    code_va_ranges = [(cr.va, cr.va + len(cr.data)) for cr in code_ranges]

    def _in_code(va: int) -> bool:
        return any(lo <= va < hi for lo, hi in code_va_ranges)

    for cr in code_ranges:
        for insn in md.disasm(cr.data, cr.va):
            all_code_insns[insn.address] = insn

    # If no .init/.text/.fini sections (e.g., section-stripped ELF), fallback to PT_LOAD PF_X
    if not code_ranges:
        for idx, ph in enumerate(phdrs):
            if ph.p_type == 1 and (ph.p_flags & 1) and ph.p_filesz > 0:
                cr = CodeRange(
                    name=f".rx_{idx}",
                    va=ph.p_vaddr,
                    offset=ph.p_offset,
                    data=raw_bytes[ph.p_offset : ph.p_offset + ph.p_filesz],
                )
                code_ranges.append(cr)
                code_va_ranges.append((cr.va, cr.va + len(cr.data)))
                for insn in md.disasm(cr.data, cr.va):
                    all_code_insns[insn.address] = insn

    # Seed function entries
    functions: Set[int] = set()
    addr_taken_funcs: Set[int] = set()
    for s in [e_entry, init_va, fini_va] + init_array + fini_array:
        if s and _in_code(s):
            functions.add(s)
    for va in sym_names:
        if _in_code(va):
            functions.add(va)
    for r in rel_dyn:
        if r.r_type == 8:  # R_386_RELATIVE
            val = _read_u32(phdrs, raw_bytes, r.offset)
            if val is not None and _in_code(val):
                functions.add(val)
                addr_taken_funcs.add(val)

    # Run angr CFGFast
    proj, cfg = _run_angr_cfg(path, verbose=verbose)
    angr_jump_tables: Dict[int, List[int]] = {}
    if cfg is not None and proj is not None:
        for faddr, func in cfg.kb.functions.items():
            va_int = int(faddr)
            if not func.is_plt and not func.is_simprocedure and _in_code(va_int):
                first_insn = all_code_insns.get(va_int)
                if first_insn is not None and first_insn.id in (
                    cs_x86.X86_INS_HLT,
                    cs_x86.X86_INS_NOP,
                    cs_x86.X86_INS_XCHG,
                ) and va_int != e_entry and va_int not in sym_names:
                    continue
                functions.add(va_int)
        if hasattr(cfg, "jump_tables"):
            for jaddr, jt in cfg.jump_tables.items():
                if hasattr(jt, "jumptable_entries") and jt.jumptable_entries:
                    angr_jump_tables[int(jaddr)] = [int(t) for t in jt.jumptable_entries]

    # Run radare2 on stripped binaries or --use-r2
    if use_r2 or not has_symtab:
        r2_funcs, _ = _run_r2_analysis(path, verbose=verbose)
        for va in r2_funcs:
            if _in_code(va):
                functions.add(va)

    # Run Ghidra headless if --use-ghidra
    if use_ghidra:
        gh_funcs, _ = _run_ghidra_analysis(path, verbose=verbose)
        for va in gh_funcs:
            if _in_code(va):
                functions.add(va)

    # Capstone direct call target sweep across code_ranges (skip PIC `call $+5` next-insn idioms)
    for va, insn in all_code_insns.items():
        if insn.id == cs_x86.X86_INS_CALL and len(insn.operands) == 1:
            op = insn.operands[0]
            if op.type == cs_x86.X86_OP_IMM:
                tgt = op.imm & 0xFFFFFFFF
                if _in_code(tgt) and tgt != va + insn.size:
                    functions.add(tgt)

    # Strip false-positive `call $+5; pop %reg` landing pads reported as functions by rizin/angr
    false_call_next_vas: Set[int] = set()
    for va, insn in all_code_insns.items():
        if insn.id == cs_x86.X86_INS_CALL and len(insn.operands) == 1:
            op = insn.operands[0]
            if op.type == cs_x86.X86_OP_IMM:
                tgt = op.imm & 0xFFFFFFFF
                if tgt == (va + insn.size) & 0xFFFFFFFF:
                    false_call_next_vas.add(tgt)
    for fva in false_call_next_vas:
        if fva != e_entry and fva not in sym_names and fva not in init_array and fva not in fini_array:
            functions.discard(fva)

    # Detect PIC thunks: function starting with `mov (%esp), %reg32; ret`
    pic_thunks: Dict[int, str] = {}
    for fva in sorted(functions):
        i0 = all_code_insns.get(fva)
        if i0 is None:
            continue
        i1 = all_code_insns.get(fva + i0.size)
        if i1 is None:
            continue
        if (
            i0.id == cs_x86.X86_INS_MOV
            and i1.id == cs_x86.X86_INS_RET
            and len(i0.operands) == 2
        ):
            op0, op1 = i0.operands[0], i0.operands[1]
            if (
                op0.type == cs_x86.X86_OP_REG
                and op1.type == cs_x86.X86_OP_MEM
                and op1.mem.base == cs_x86.X86_REG_ESP
                and op1.mem.index == cs_x86.X86_REG_INVALID
                and op1.mem.disp == 0
            ):
                rname = _cs_reg_to_32_name(i0, op0.reg)
                if rname:
                    pic_thunks[fva] = rname

    # Recover Jump Tables by inspecting blocks ending in indirect `jmp *...`
    jump_tables: Dict[int, JumpTableInfo] = {}
    sorted_vas = sorted(all_code_insns.keys())
    va_to_idx = {va: idx for idx, va in enumerate(sorted_vas)}

    def _infer_bound(jmp_idx: int, idx_reg: int) -> Optional[int]:
        # Forward value-numbering over [max(0, jmp_idx - 16), jmp_idx]
        start_k = max(0, jmp_idx - 16)
        val_id: Dict[int, int] = {}
        next_vid = 10000

        def alloc_val() -> int:
            nonlocal next_vid
            v = next_vid
            next_vid += 1
            return v

        bound_by_val: Dict[int, int] = {}

        for k in range(start_k, jmp_idx):
            pi = all_code_insns[sorted_vas[k]]

            # On mov/movzx/movsx dst_reg, src_reg: val_id[dst_reg] = val_id[src_reg]
            if (
                pi.id in (cs_x86.X86_INS_MOV, cs_x86.X86_INS_MOVZX, cs_x86.X86_INS_MOVSX)
                and len(pi.operands) == 2
                and pi.operands[0].type == cs_x86.X86_OP_REG
                and pi.operands[1].type == cs_x86.X86_OP_REG
            ):
                dst = pi.operands[0].reg
                src = pi.operands[1].reg
                val_id[dst] = val_id.get(src, src)
                continue

            # On cmp reg, imm: record bound_by_val[val_id[reg]] = (imm & 0xFFFFFFFF) + 1
            if (
                pi.id == cs_x86.X86_INS_CMP
                and len(pi.operands) == 2
                and pi.operands[0].type == cs_x86.X86_OP_REG
                and pi.operands[1].type == cs_x86.X86_OP_IMM
            ):
                reg = pi.operands[0].reg
                bound_by_val[val_id.get(reg, reg)] = int(pi.operands[1].imm & 0xFFFFFFFF) + 1
                continue

            # On and reg, imm where mask = imm & 0xFFFFFFFF is 2^k - 1 <= 255:
            # assign new val_id[reg] and record bound_by_val[new_id] = mask + 1
            if (
                pi.id == cs_x86.X86_INS_AND
                and len(pi.operands) == 2
                and pi.operands[0].type == cs_x86.X86_OP_REG
                and pi.operands[1].type == cs_x86.X86_OP_IMM
            ):
                mask = int(pi.operands[1].imm & 0xFFFFFFFF)
                if mask > 0 and (mask & (mask + 1)) == 0 and mask <= 255:
                    new_id = alloc_val()
                    val_id[pi.operands[0].reg] = new_id
                    bound_by_val[new_id] = mask + 1
                    continue

            # On call imm: if target == pi.address + pi.size (inline call $+5) do NOT clobber any registers;
            # if target in pic_thunks, only invalidate that thunk's target register;
            # otherwise invalidate caller-saved EAX, ECX, EDX.
            if pi.id == cs_x86.X86_INS_CALL:
                if len(pi.operands) == 1 and pi.operands[0].type == cs_x86.X86_OP_IMM:
                    target = pi.operands[0].imm & 0xFFFFFFFF
                    if target == (pi.address + pi.size) & 0xFFFFFFFF:
                        continue
                    if target in pic_thunks:
                        thunk_rname = pic_thunks[target]
                        thunk_reg = {
                            "eax": cs_x86.X86_REG_EAX,
                            "ecx": cs_x86.X86_REG_ECX,
                            "edx": cs_x86.X86_REG_EDX,
                            "ebx": cs_x86.X86_REG_EBX,
                            "esp": cs_x86.X86_REG_ESP,
                            "ebp": cs_x86.X86_REG_EBP,
                            "esi": cs_x86.X86_REG_ESI,
                            "edi": cs_x86.X86_REG_EDI,
                        }.get(thunk_rname)
                        if thunk_reg is not None:
                            val_id[thunk_reg] = alloc_val()
                        continue
                for reg in (cs_x86.X86_REG_EAX, cs_x86.X86_REG_ECX, cs_x86.X86_REG_EDX):
                    val_id[reg] = alloc_val()
                continue

            # On any other instruction modifying operands[0] when it is a register
            # (excluding CMP, TEST, PUSH, JMP, Jcc): assign a fresh val_id to operands[0].reg
            if (
                pi.id in (cs_x86.X86_INS_CMP, cs_x86.X86_INS_TEST, cs_x86.X86_INS_PUSH)
                or pi.mnemonic.startswith("j")
            ):
                continue

            if len(pi.operands) > 0 and pi.operands[0].type == cs_x86.X86_OP_REG:
                val_id[pi.operands[0].reg] = alloc_val()

        return bound_by_val.get(val_id.get(idx_reg, idx_reg))

    for va in sorted_vas:
        insn = all_code_insns[va]
        if insn.id != cs_x86.X86_INS_JMP or len(insn.operands) != 1:
            continue
        op0 = insn.operands[0]
        if op0.type == cs_x86.X86_OP_IMM:
            continue

        jidx = va_to_idx[va]
        # Pattern 1: Non-PIE `jmp *tbl_va(,%idx,4)`
        if op0.type == cs_x86.X86_OP_MEM and op0.mem.base == cs_x86.X86_REG_INVALID and op0.mem.scale == 4:
            tbl_va = op0.mem.disp & 0xFFFFFFFF
            idx_reg = op0.mem.index
            bound = _infer_bound(jidx, idx_reg)
            if bound is None and va in angr_jump_tables:
                bound = len(angr_jump_tables[va])
            if bound is None:
                # Probe consecutive valid code targets
                bound = 0
                while bound < 256:
                    t = _read_u32(phdrs, raw_bytes, tbl_va + bound * 4)
                    if t is None or not _in_code(t):
                        break
                    bound += 1
            if bound and bound > 0:
                targets = []
                for i in range(bound):
                    t = _read_u32(phdrs, raw_bytes, tbl_va + i * 4)
                    if t is not None and _in_code(t):
                        targets.append(t)
                if len(targets) == bound:
                    jump_tables[va] = JumpTableInfo(
                        jmp_va=va,
                        replace_start_va=va,
                        replace_end_va=va + insn.size,
                        tbl_va=tbl_va,
                        is_pic=False,
                        got_reg=None,
                        idx_reg=idx_reg,
                        targets=targets,
                    )
                    continue

        # Pattern 2 & 3: PIC `jmp *reg` preceded by:
        #   add disp(%got, %idx, 4), %got
        # or:
        #   mov disp(%got, %idx, 4), %tmp; add %got, %tmp
        if op0.type == cs_x86.X86_OP_REG and jidx >= 1:
            jmp_reg = op0.reg
            prev1 = all_code_insns[sorted_vas[jidx - 1]]
            # Pattern 2: prev1 is `add disp(%got, %idx, 4), %got`
            if (
                prev1.id == cs_x86.X86_INS_ADD
                and len(prev1.operands) == 2
                and prev1.operands[0].type == cs_x86.X86_OP_REG
                and prev1.operands[0].reg == jmp_reg
                and prev1.operands[1].type == cs_x86.X86_OP_MEM
                and prev1.operands[1].mem.base == jmp_reg
                and prev1.operands[1].mem.index != cs_x86.X86_REG_INVALID
                and prev1.operands[1].mem.scale == 4
            ):
                got_reg = jmp_reg
                idx_reg = prev1.operands[1].mem.index
                tbl_va = (got_va + prev1.operands[1].mem.disp) & 0xFFFFFFFF
                bound = _infer_bound(jidx - 1, idx_reg)
                if bound is None and va in angr_jump_tables:
                    bound = len(angr_jump_tables[va])
                if bound is None:
                    # Probe consecutive valid code targets
                    bound = 0
                    while bound < 256:
                        rel = _read_i32(phdrs, raw_bytes, tbl_va + bound * 4)
                        if rel is None:
                            break
                        t = (got_va + rel) & 0xFFFFFFFF
                        if not _in_code(t):
                            break
                        bound += 1
                targets = []
                if bound and bound > 0:
                    for i in range(bound):
                        rel = _read_i32(phdrs, raw_bytes, tbl_va + i * 4)
                        if rel is not None:
                            t = (got_va + rel) & 0xFFFFFFFF
                            if _in_code(t):
                                targets.append(t)
                if bound and len(targets) == bound:
                    jump_tables[va] = JumpTableInfo(
                        jmp_va=va,
                        replace_start_va=prev1.address,
                        replace_end_va=va + insn.size,
                        tbl_va=tbl_va,
                        is_pic=True,
                        got_reg=got_reg,
                        idx_reg=idx_reg,
                        targets=targets,
                    )
                    continue

            # Pattern 3: prev2 `mov disp(%got, %idx, 4), %tmp`, prev1 `add %got, %tmp`, `jmp *%tmp`
            if jidx >= 2:
                prev2 = all_code_insns[sorted_vas[jidx - 2]]
                if (
                    prev1.id == cs_x86.X86_INS_ADD
                    and len(prev1.operands) == 2
                    and prev1.operands[0].type == cs_x86.X86_OP_REG
                    and prev1.operands[0].reg == jmp_reg
                    and prev1.operands[1].type == cs_x86.X86_OP_REG
                    and prev2.id == cs_x86.X86_INS_MOV
                    and len(prev2.operands) == 2
                    and prev2.operands[0].type == cs_x86.X86_OP_REG
                    and prev2.operands[0].reg == jmp_reg
                    and prev2.operands[1].type == cs_x86.X86_OP_MEM
                    and prev2.operands[1].mem.base == prev1.operands[1].reg
                    and prev2.operands[1].mem.index != cs_x86.X86_REG_INVALID
                    and prev2.operands[1].mem.scale == 4
                ):
                    got_reg = prev1.operands[1].reg
                    idx_reg = prev2.operands[1].mem.index
                    tbl_va = (got_va + prev2.operands[1].mem.disp) & 0xFFFFFFFF
                    bound = _infer_bound(jidx - 2, idx_reg)
                    if bound is None and va in angr_jump_tables:
                        bound = len(angr_jump_tables[va])
                    if bound is None:
                        # Probe consecutive valid code targets
                        bound = 0
                        while bound < 256:
                            rel = _read_i32(phdrs, raw_bytes, tbl_va + bound * 4)
                            if rel is None:
                                break
                            t = (got_va + rel) & 0xFFFFFFFF
                            if not _in_code(t):
                                break
                            bound += 1
                    targets = []
                    if bound and bound > 0:
                        for i in range(bound):
                            rel = _read_i32(phdrs, raw_bytes, tbl_va + i * 4)
                            if rel is not None:
                                t = (got_va + rel) & 0xFFFFFFFF
                                if _in_code(t):
                                    targets.append(t)
                    if bound and len(targets) == bound:
                        jump_tables[va] = JumpTableInfo(
                            jmp_va=va,
                            replace_start_va=prev2.address,
                            replace_end_va=va + insn.size,
                            tbl_va=tbl_va,
                            is_pic=True,
                            got_reg=got_reg,
                            idx_reg=idx_reg,
                            targets=targets,
                        )
                        continue

    # Reachability traversal (CFG worklist from all known entry points & functions)
    # This validates every reachable instruction against N07/N08 and identifies syscalls.
    visited_insns: Dict[int, capstone.CsInsn] = {}
    syscalls: Dict[int, str] = {}
    unresolved_indirect_jmps: Set[int] = set()

    def _is_noreturn_call(insn: capstone.CsInsn) -> bool:
        if insn.id != cs_x86.X86_INS_CALL or len(insn.operands) != 1:
            return False
        op = insn.operands[0]
        if op.type == cs_x86.X86_OP_IMM:
            tgt = op.imm & 0xFFFFFFFF
            stub = plt_stubs.get(tgt)
            if stub and stub.sym_name in NORETURN_SYMS:
                return True
            sname = sym_names.get(tgt, "")
            if sname in NORETURN_SYMS:
                return True
        return False

    # Also detect address-taken functions via `lea disp(%got_reg), %reg` or `push/mov $imm`
    tracker = GotProvenanceTracker(got_va, pic_thunks, functions)
    sorted_all_vas = sorted(all_code_insns.keys())
    for s_i, va in enumerate(sorted_all_vas):
        insn = all_code_insns[va]
        next_insn = all_code_insns.get(sorted_all_vas[s_i + 1]) if s_i + 1 < len(sorted_all_vas) else None
        if insn.id == cs_x86.X86_INS_LEA and len(insn.operands) == 2:
            op1 = insn.operands[1]
            if (
                op1.type == cs_x86.X86_OP_MEM
                and op1.mem.index == cs_x86.X86_REG_INVALID
                and tracker.is_got_reg(op1.mem.base)
            ):
                cand = (got_va + op1.mem.disp) & 0xFFFFFFFF
                if _in_code(cand) and cand in all_code_insns:
                    functions.add(cand)
                    addr_taken_funcs.add(cand)
        elif e_type == 2:  # ET_EXEC immediate function pointer
            for op in insn.operands:
                if op.type == cs_x86.X86_OP_IMM:
                    cand = op.imm & 0xFFFFFFFF
                    if cand in functions:
                        addr_taken_funcs.add(cand)
        tracker.on_instruction(insn, next_insn)

    for s in init_array + fini_array:
        if _in_code(s):
            addr_taken_funcs.add(s)

    worklist: List[int] = sorted(functions)
    seen_va: Set[int] = set()
    while worklist:
        cur_va = worklist.pop()
        if cur_va in seen_va or not _in_code(cur_va):
            continue

        # Disassemble linearly along current basic block
        cr = next((c for c in code_ranges if c.va <= cur_va < c.va + len(c.data)), None)
        if cr is None:
            continue
        off = cur_va - cr.va
        for insn in md.disasm(cr.data[off:], cur_va):
            if insn.address in seen_va:
                break
            seen_va.add(insn.address)
            visited_insns[insn.address] = insn

            bad_reason = _is_forbidden_insn(insn)
            if bad_reason is not None:
                hex_b = " ".join(f"{b:02x}" for b in insn.bytes)
                raise PatcherError(
                    f"'{path}': {bad_reason} at guest addr 0x{insn.address:08x} (bytes: {hex_b})"
                )

            if insn.id == cs_x86.X86_INS_INT and len(insn.bytes) >= 2 and insn.bytes[1] == 0x80:
                syscalls[insn.address] = "int80"
            elif insn.id == cs_x86.X86_INS_SYSENTER:
                syscalls[insn.address] = "sysenter"

            if insn.id == cs_x86.X86_INS_RET or insn.id == cs_x86.X86_INS_RETF:
                break

            if insn.id == cs_x86.X86_INS_JMP:
                if len(insn.operands) == 1 and insn.operands[0].type == cs_x86.X86_OP_IMM:
                    tgt = insn.operands[0].imm & 0xFFFFFFFF
                    if _in_code(tgt) and tgt not in seen_va:
                        worklist.append(tgt)
                elif insn.address in jump_tables:
                    for tgt in jump_tables[insn.address].targets:
                        if tgt not in seen_va:
                            worklist.append(tgt)
                else:
                    # Check if jmp *got_slot (PLT tail)
                    op0 = insn.operands[0]
                    is_plt_jmp = False
                    if insn.address in plt_stubs:
                        is_plt_jmp = True
                    elif (
                        op0.type == cs_x86.X86_OP_MEM
                        and op0.mem.index == cs_x86.X86_REG_INVALID
                    ):
                        slot_va = None
                        if op0.mem.base == cs_x86.X86_REG_INVALID:
                            slot_va = op0.mem.disp & 0xFFFFFFFF
                        elif got_va != 0 and op0.mem.base == cs_x86.X86_REG_EBX:
                            slot_va = (got_va + op0.mem.disp) & 0xFFFFFFFF
                        if slot_va is not None:
                            rel = jmp_slot_by_got.get(slot_va) or glob_dat_by_got.get(slot_va)
                            if rel is not None and rel.sym_name:
                                plt_stubs.setdefault(
                                    insn.address,
                                    PltStubInfo(
                                        va=insn.address,
                                        size=insn.size,
                                        sym_name=rel.sym_name,
                                        got_slot_va=slot_va,
                                    ),
                                )
                                is_plt_jmp = True
                    if not is_plt_jmp:
                        unresolved_indirect_jmps.add(insn.address)
                break

            if insn.id in (
                cs_x86.X86_INS_JA, cs_x86.X86_INS_JAE, cs_x86.X86_INS_JB, cs_x86.X86_INS_JBE,
                cs_x86.X86_INS_JCXZ, cs_x86.X86_INS_JECXZ, cs_x86.X86_INS_JE, cs_x86.X86_INS_JG,
                cs_x86.X86_INS_JGE, cs_x86.X86_INS_JL, cs_x86.X86_INS_JLE, cs_x86.X86_INS_JNE,
                cs_x86.X86_INS_JNO, cs_x86.X86_INS_JNP, cs_x86.X86_INS_JNS, cs_x86.X86_INS_JO,
                cs_x86.X86_INS_JP, cs_x86.X86_INS_JS, cs_x86.X86_INS_LOOP, cs_x86.X86_INS_LOOPE,
                cs_x86.X86_INS_LOOPNE,
            ):
                if len(insn.operands) == 1 and insn.operands[0].type == cs_x86.X86_OP_IMM:
                    tgt = insn.operands[0].imm & 0xFFFFFFFF
                    if _in_code(tgt) and tgt not in seen_va:
                        worklist.append(tgt)
                continue

            if insn.id == cs_x86.X86_INS_CALL:
                if len(insn.operands) == 1 and insn.operands[0].type == cs_x86.X86_OP_IMM:
                    tgt = insn.operands[0].imm & 0xFFFFFFFF
                    if _in_code(tgt) and tgt not in seen_va:
                        worklist.append(tgt)
                if _is_noreturn_call(insn):
                    break

        # Check if disassembly stopped early on invalid/undecodable bytes inside reachable block
        else:
            pass

    # If the very first instruction at e_entry failed to decode (invalid opcode bytes)
    if e_entry != 0 and _in_code(e_entry) and e_entry not in visited_insns:
        b = _vaddr_to_bytes(phdrs, raw_bytes, e_entry, 4)
        raise PatcherError(
            f"'{path}': unable to decode i386 instruction at e_entry=0x{e_entry:08x} (bytes={b.hex() if b else '?'})"
        )

    # Ensure every instruction in .init/.text/.fini that isn't dead padding after noreturn/jmp/ret is also checked
    # Classify functions
    sorted_funcs = sorted(functions)
    func_classifications: Dict[int, str] = {}
    for idx, fva in enumerate(sorted_funcs):
        f_end = sorted_funcs[idx + 1] if idx + 1 < len(sorted_funcs) else 0xFFFFFFFF
        needs_dbt = any(fva <= jva < f_end for jva in unresolved_indirect_jmps)
        func_classifications[fva] = "REQUIRES_DBT_FALLBACK" if needs_dbt else "REWRITABLE_NATIVE"

    return AnalysisResult(
        path=path,
        raw_bytes=raw_bytes,
        e_type=e_type,
        e_entry=e_entry,
        has_interp=has_interp,
        interp_str=interp_str,
        has_symtab=has_symtab,
        phdrs=phdrs,
        shdrs=shdrs,
        dyn_tags=dyn_tags,
        dt_needed=dt_needed,
        dynsyms=dynsyms,
        rel_dyn=rel_dyn,
        rel_plt=rel_plt,
        got_va=got_va,
        got_plt_va=got_plt_va,
        plt_stubs=plt_stubs,
        code_ranges=code_ranges,
        init_va=init_va,
        fini_va=fini_va,
        init_array=init_array,
        fini_array=fini_array,
        tls_info=tls_info,
        functions=functions,
        sym_names=sym_names,
        pic_thunks=pic_thunks,
        jump_tables=jump_tables,
        syscalls=syscalls,
        addr_taken_funcs=addr_taken_funcs,
        reachable_insns=visited_insns,
        func_classifications=func_classifications,
        max_vaddr=max_vaddr,
        symbol_versions=symbol_versions,
        sym_version_map=sym_version_map,
    )
