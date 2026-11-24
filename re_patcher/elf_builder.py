"""Stage 2c — EM_X86_64 ELF32 (x32) Builder: DT_RELA, .dynsym/.hash, PT_INTERP, .text.newminhh, .note.newminhh."""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Set, Tuple

from . import PatcherError
from .analyzer import AnalysisResult, DynSymInfo, PhdrInfo, RelocInfo
from .lookup import (
    LookupSymEntry,
    LookupTable,
    ParsedSig,
    NM_RET_VOID,
    NM_SYM_FLAG_DATA,
    NM_SYM_FLAG_IFUNC,
    NM_SYM_FLAG_NATIVE,
    NM_SYM_FLAG_PATCHED,
    NM_SYM_FLAG_WEAK,
)
from .rewriter import RewrittenOutput, rewrite_binary
from .trampolines import (
    assemble_and_link_helpers,
    generate_helper_assembly,
    resolve_import_binding,
)

EM_X86_64 = 62
DEFAULT_X32_INTERP = "/libx32/ld-linux-x32.so.2"

PT_LOAD = 1
PT_DYNAMIC = 2
PT_INTERP = 3
PT_NOTE = 4
PT_PHDR = 6
PT_TLS = 7
PT_GNU_PROPERTY = 0x6474E553

PF_X = 1
PF_W = 2
PF_R = 4

# x86_64 / x32 relocation types
R_X86_64_NONE = 0
R_X86_64_64 = 1
R_X86_64_PC32 = 2
R_X86_64_COPY = 5
R_X86_64_GLOB_DAT = 6
R_X86_64_JUMP_SLOT = 7
R_X86_64_RELATIVE = 8
R_X86_64_32 = 10
R_X86_64_DTPMOD64 = 16
R_X86_64_DTPOFF32 = 21
R_X86_64_TPOFF32 = 23
R_X86_64_IRELATIVE = 37

# Dynamic tags
DT_NULL = 0
DT_NEEDED = 1
DT_PLTRELSZ = 2
DT_PLTGOT = 3
DT_HASH = 4
DT_STRTAB = 5
DT_SYMTAB = 6
DT_RELA = 7
DT_RELASZ = 8
DT_RELAENT = 9
DT_STRSZ = 10
DT_SYMENT = 11
DT_INIT = 12
DT_FINI = 13
DT_SONAME = 14
DT_PLTREL = 20
DT_DEBUG = 21
DT_JMPREL = 23
DT_BIND_NOW = 24
DT_INIT_ARRAY = 25
DT_FINI_ARRAY = 26
DT_INIT_ARRAYSZ = 27
DT_FINI_ARRAYSZ = 28
DT_FLAGS = 30
DT_FLAGS_1 = 0x6FFFFFFB
DT_VERSYM = 0x6FFFFFF0
DT_VERNEED = 0x6FFFFFFE
DT_VERNEEDNUM = 0x6FFFFFFF

# Section types
SHT_GNU_verneed = 0x6FFFFFFE
SHT_GNU_versym = 0x6FFFFFFF

# .note.newminhh v2 constants matching newminhh_abi.h
NM_NOTE_NAME = b"newminhh\x00"
NM_NOTE_TYPE_PATCHED = 0x4E4D4848
NM_NOTE_V2_VERSION = 2
NM_NOTE_V2_FLAG_CDECL_ALIASES = 1 << 0
NM_NOTE_V2_FLAG_LEVEL_SHIFT = 4


def _align_up(val: int, align: int) -> int:
    if align <= 1:
        return val
    return (val + align - 1) & ~(align - 1)


def _elf_hash(name: str) -> int:
    h = 0
    for b in name.encode("ascii", errors="replace"):
        h = ((h << 4) + b) & 0xFFFFFFFF
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFF
    return h


def _build_sysv_hash(names: List[str]) -> bytes:
    nchain = len(names)
    nbucket = max(1, nchain | 1)
    buckets = [0] * nbucket
    chains = [0] * nchain
    for idx in range(1, nchain):
        name = names[idx]
        if not name:
            continue
        b = _elf_hash(name) % nbucket
        if buckets[b] == 0:
            buckets[b] = idx
        else:
            cur = buckets[b]
            while chains[cur] != 0:
                cur = chains[cur]
            chains[cur] = idx
    out = bytearray()
    out.extend(struct.pack("<II", nbucket, nchain))
    for v in buckets:
        out.extend(struct.pack("<I", v))
    for v in chains:
        out.extend(struct.pack("<I", v))
    return bytes(out)


def _build_note_newminhh(hybrid_mode: str, level: int = 1) -> bytes:
    """Build .note.newminhh v2 descriptor (type 0x4E4D4848, version=2, flags)."""
    name = NM_NOTE_NAME
    flags = NM_NOTE_V2_FLAG_CDECL_ALIASES | ((level & 0xF) << NM_NOTE_V2_FLAG_LEVEL_SHIFT)
    desc = struct.pack("<II", NM_NOTE_V2_VERSION, flags)
    name_pad = name + b"\x00" * ((4 - len(name) % 4) % 4)
    desc_pad = desc + b"\x00" * ((4 - len(desc) % 4) % 4)
    return struct.pack("<III", len(name), len(desc), NM_NOTE_TYPE_PATCHED) + name_pad + desc_pad


def _vaddr_to_offset(phdrs: List[PhdrInfo], va: int) -> Optional[int]:
    for ph in phdrs:
        if ph.p_type == PT_LOAD and ph.p_filesz > 0:
            if ph.p_vaddr <= va < ph.p_vaddr + ph.p_filesz:
                return ph.p_offset + (va - ph.p_vaddr)
    return None


def _read_u32_at_va(phdrs: List[PhdrInfo], raw: bytes, va: int) -> int:
    off = _vaddr_to_offset(phdrs, va)
    if off is None or off + 4 > len(raw):
        return 0
    return struct.unpack_from("<I", raw, off)[0]


def build_x32_elf(
    analysis: AnalysisResult,
    initial_rewritten: RewrittenOutput,
    hybrid_mode: str = "auto",
    verbose: bool = False,
    lookup_table: Optional[LookupTable] = None,
    target_interp: Optional[str] = None,
    allow_untyped_unsafe: bool = False,
    level: int = 1,
    boost_report: Optional[Any] = None,
) -> bytes:
    """Construct final EM_X86_64 ELF32 binary/SO with rewritten .text.newminhh and DT_RELA."""
    if lookup_table is None:
        lookup_table = LookupTable.load()

    need_dbt = (hybrid_mode == "hybrid") or (len(initial_rewritten.fallback_funcs) > 0)
    fallback_va_set = {va for _lbl, va in initial_rewritten.fallback_funcs}
    use_dbt_embedded_entry = (
        (hybrid_mode == "hybrid")
        and (analysis.e_type == 3)
        and (analysis.e_entry != 0)
        and (analysis.e_entry in fallback_va_set)
    )

    # Resolve interpreter path
    if target_interp is not None:
        interp_path = target_interp
    elif lookup_table is not None and lookup_table.interp:
        interp_path = lookup_table.interp
    else:
        interp_path = DEFAULT_X32_INTERP
    interp_bytes = (interp_path.encode("ascii") + b"\x00") if analysis.has_interp else b""

    # 1. Resolve all imported symbols via lookup_table
    resolved_imports: Dict[str, LookupSymEntry] = {}
    for sym in analysis.dynsyms:
        if sym.st_shndx == 0 and sym.name:
            if sym.name.startswith("__nm_ifunc_"):
                ent = LookupSymEntry(
                    soname="",
                    sym=sym.name,
                    tgt_soname="",
                    tgt_sym=sym.name,
                    tgt_ver=None,
                    cls="PATCHED_CDECL",
                    sig_str="",
                    parsed_sig=None,
                    flags=0,
                    raw_flags="",
                    size=0,
                )
            else:
                orig_soname, orig_ver = analysis.sym_version_map.get(sym.name, (None, None))
                ent = resolve_import_binding(
                    lookup_table,
                    sym.name,
                    orig_soname,
                    orig_ver,
                    allow_untyped_unsafe=allow_untyped_unsafe,
                    dt_needed=analysis.dt_needed,
                )
            resolved_imports[sym.name] = ent

    # Also recognize synthetic IFUNC PLT entries in resolved_imports
    for orig_sym, _gva, _sidx in initial_rewritten.plt_entries:
        if orig_sym.startswith("__nm_ifunc_") and orig_sym not in resolved_imports:
            resolved_imports[orig_sym] = LookupSymEntry(
                soname="",
                sym=orig_sym,
                tgt_soname="",
                tgt_sym=orig_sym,
                tgt_ver=None,
                cls="PATCHED_CDECL",
                sig_str="",
                parsed_sig=None,
                flags=0,
                raw_flags="",
                size=0,
            )

    # Verify R_386_COPY st_size parity
    for r in analysis.rel_dyn:
        if r.r_type == 5:  # R_386_COPY
            if r.sym_idx < len(analysis.dynsyms):
                sym = analysis.dynsyms[r.sym_idx]
                ent = resolved_imports.get(sym.name)
                if ent is not None:
                    if ent.size != 0 and sym.st_size != 0 and ent.size != sym.st_size:
                        raise PatcherError(
                            f"R_386_COPY size mismatch for symbol '{sym.name}': "
                            f"lookup table expects {ent.size} bytes, ELF dynsym defines {sym.st_size} bytes"
                        )

    # 2. Build initial dynsym list & collect exported functions
    new_dynsyms: List[DynSymInfo] = []
    sym_name_to_idx: Dict[str, int] = {}
    exported_funcs: List[Tuple[str, int, int]] = []  # (sym_name, orig_va, dynsym_idx)

    for idx, sym in enumerate(analysis.dynsyms):
        mapped_name = sym.name
        if sym.st_shndx == 0 and sym.name:
            ent = resolved_imports.get(sym.name)
            if ent is not None and ent.tgt_sym:
                mapped_name = ent.tgt_sym
        ns = DynSymInfo(
            index=idx,
            name=mapped_name,
            st_value=sym.st_value,
            st_size=sym.st_size,
            st_info=sym.st_info,
            st_other=sym.st_other,
            st_shndx=sym.st_shndx,
        )
        new_dynsyms.append(ns)
        if mapped_name:
            sym_name_to_idx.setdefault(mapped_name, idx)
        if sym.st_shndx != 0 and sym.bind in (1, 2) and sym.sym_type == 2 and sym.st_value != 0:
            exported_funcs.append((sym.name, sym.st_value, idx))

    if not new_dynsyms:
        new_dynsyms.append(DynSymInfo(0, "", 0, 0, 0, 0, 0))

    # For ET_DYN, also register __nm_cdecl.<sym> in .dynsym pointing directly to cdecl code
    if analysis.e_type == 3:  # ET_DYN
        for sname, sva, didx in exported_funcs:
            cdecl_alias_name = f"__nm_cdecl.{sname}"
            if cdecl_alias_name not in sym_name_to_idx:
                nidx = len(new_dynsyms)
                new_dynsyms.append(
                    DynSymInfo(
                        index=nidx,
                        name=cdecl_alias_name,
                        st_value=0,  # updated after rewrite
                        st_size=new_dynsyms[didx].st_size,
                        st_info=(1 << 4) | 2,  # STB_GLOBAL | STT_FUNC
                        st_other=new_dynsyms[didx].st_other,
                        st_shndx=1,  # updated to .text.newminhh section index later
                    )
                )
                sym_name_to_idx[cdecl_alias_name] = nidx

    # Extra runtime imports
    extra_imports = [
        "newminhh_compat_syscall",
        "newminhh_call_x32_cdecl",
        "newminhh_invoke_cdecl_asm",
        "newminhh_get_stdio_streams",
        "newminhh_compat_fixup_copy_data",
        "newminhh_out_common",
        "newminhh_register_native_cdecl_range",
        "newminhh_unregister_native_cdecl_range",
        "newminhh_is_native_cdecl_va",
        "compat_atexit",
    ]
    if need_dbt:
        extra_imports.extend(["newminhh_dbt_enter_at", "newminhh_dbt_exec_embedded"])

    extra_got_syms: List[str] = []
    for imp_name in extra_imports:
        if imp_name not in sym_name_to_idx:
            nidx = len(new_dynsyms)
            new_dynsyms.append(
                DynSymInfo(
                    index=nidx,
                    name=imp_name,
                    st_value=0,
                    st_size=0,
                    st_info=(1 << 4) | 2,  # STB_GLOBAL | STT_FUNC
                    st_other=0,
                    st_shndx=0,
                )
            )
            sym_name_to_idx[imp_name] = nidx
        extra_got_syms.append(imp_name)

    extra_plt_got_slots: Dict[int, Tuple[int, str]] = {}
    for idx, (orig_sym, got_slot_va, _sidx) in enumerate(initial_rewritten.plt_entries):
        if got_slot_va == 0:
            ent = resolved_imports.get(orig_sym)
            mapped_name = ent.tgt_sym if (ent and ent.tgt_sym) else orig_sym
            if mapped_name not in sym_name_to_idx:
                nidx = len(new_dynsyms)
                new_dynsyms.append(
                    DynSymInfo(
                        index=nidx,
                        name=mapped_name,
                        st_value=0,
                        st_size=0,
                        st_info=(1 << 4) | 2,  # STB_GLOBAL | STT_FUNC
                        st_other=0,
                        st_shndx=0,
                    )
                )
                sym_name_to_idx[mapped_name] = nidx
            slot_offset_idx = len(extra_got_syms)
            extra_got_syms.append(mapped_name)
            extra_plt_got_slots[idx] = (slot_offset_idx, mapped_name)

    # 3. Dynamic DT_NEEDED union:
    # {"libnewminhh_compat.so"} (+ {"libnewminhh_dbt.so"} when DBT needed)
    # UNION all entry.tgt_soname from resolved imports
    needed_libs: List[str] = ["libnewminhh_compat.so"]
    if need_dbt:
        needed_libs.append("libnewminhh_dbt.so")

    for _sname, ent in resolved_imports.items():
        if ent.tgt_soname and ent.tgt_soname != "-" and ent.tgt_soname not in needed_libs:
            needed_libs.append(ent.tgt_soname)

    # 4. Build .dynstr and .hash
    dynstr = bytearray(b"\x00")
    str_offsets: Dict[str, int] = {"": 0}

    def _add_dynstr(s: str) -> int:
        if s in str_offsets:
            return str_offsets[s]
        off = len(dynstr)
        dynstr.extend(s.encode("ascii", errors="replace") + b"\x00")
        str_offsets[s] = off
        return off

    for s in new_dynsyms:
        _add_dynstr(s.name)
    needed_str_offs = [_add_dynstr(lib) for lib in needed_libs]

    soname_str = ""
    if 14 in analysis.dyn_tags:  # DT_SONAME
        orig_strtab = b""
        if 5 in analysis.dyn_tags and 10 in analysis.dyn_tags:
            soff = _vaddr_to_offset(analysis.phdrs, analysis.dyn_tags[5])
            if soff is not None:
                orig_strtab = analysis.raw_bytes[soff : soff + analysis.dyn_tags[10]]
        soname_off = analysis.dyn_tags[14]
        if soname_off < len(orig_strtab):
            end = orig_strtab.find(b"\x00", soname_off)
            soname_str = orig_strtab[soname_off : end if end >= 0 else len(orig_strtab)].decode("ascii", errors="replace")
    soname_dyn_off = _add_dynstr(soname_str) if soname_str else 0

    hash_bytes = _build_sysv_hash([s.name for s in new_dynsyms])
    dynsym_size = len(new_dynsyms) * 16

    # 5. ELF Symbol Versioning Synthesis (.gnu.version / .gnu.version_r)
    # Check if ANY resolved import has a pinned tgt_ver
    has_pinned_ver = any(ent.tgt_ver is not None for ent in resolved_imports.values())
    grouped_verneeds: Dict[str, Dict[str, int]] = {}
    verneed_bytes = b""
    versym_bytes = b""
    num_verneed = 0

    if has_pinned_ver:
        next_ver_idx = 2
        for _sname, ent in sorted(resolved_imports.items()):
            if ent.tgt_ver:
                so = ent.tgt_soname or "libnewminhh_compat.so"
                if so not in grouped_verneeds:
                    grouped_verneeds[so] = {}
                if ent.tgt_ver not in grouped_verneeds[so]:
                    grouped_verneeds[so][ent.tgt_ver] = next_ver_idx
                    next_ver_idx += 1

        # Synthesize .gnu.version_r (Elf32_Verneed / Elf32_Vernaux)
        verneed_buf = bytearray()
        so_list = list(grouped_verneeds.keys())
        for so_idx, so in enumerate(so_list):
            ver_dict = grouped_verneeds[so]
            is_last_so = (so_idx == len(so_list) - 1)
            vn_version = 1  # VER_NEED_CURRENT
            vn_cnt = len(ver_dict)
            vn_file = _add_dynstr(so)
            vn_aux = 16  # sizeof(Elf32_Verneed)
            vn_next = 0 if is_last_so else (16 + vn_cnt * 16)
            verneed_buf.extend(struct.pack("<HHIII", vn_version, vn_cnt, vn_file, vn_aux, vn_next))

            ver_items = list(ver_dict.items())
            for v_idx, (v_name, v_num) in enumerate(ver_items):
                is_last_v = (v_idx == len(ver_items) - 1)
                vna_hash = _elf_hash(v_name)
                vna_flags = 0
                vna_other = v_num
                vna_name = _add_dynstr(v_name)
                vna_next = 0 if is_last_v else 16
                verneed_buf.extend(struct.pack("<IHHII", vna_hash, vna_flags, vna_other, vna_name, vna_next))

        verneed_bytes = bytes(verneed_buf)
        num_verneed = len(grouped_verneeds)

        # Synthesize .gnu.version (Elf32_Half per dynsym)
        versym_buf = bytearray()
        for s in new_dynsyms:
            if s.index == 0:
                val = 0
            elif s.st_shndx == 0:
                ent = resolved_imports.get(s.name)
                if ent and ent.tgt_ver:
                    so = ent.tgt_soname or "libnewminhh_compat.so"
                    val = grouped_verneeds.get(so, {}).get(ent.tgt_ver, 1)
                else:
                    val = 1
            else:
                val = 1
            versym_buf.extend(struct.pack("<H", val))
        versym_bytes = bytes(versym_buf)

    dynstr_bytes = bytes(dynstr)

    # 6. Compute sizes for ext_ro_meta (Segment 1)
    kept_orig_phdrs = [p for p in analysis.phdrs if p.p_type != PT_GNU_PROPERTY]
    has_pt_phdr = any(p.p_type == PT_PHDR for p in kept_orig_phdrs)
    new_phnum = len(kept_orig_phdrs) + (0 if has_pt_phdr else 0) + 5  # 4 PT_LOADs + 1 PT_NOTE
    phdr_table_sz = new_phnum * 32

    rela_dyn_count = len(analysis.rel_dyn) + len(extra_got_syms)
    rela_dyn_sz = rela_dyn_count * 12
    rela_plt_sz = len(analysis.rel_plt) * 12
    note_bytes = _build_note_newminhh(hybrid_mode, level=level)

    # Omit .newminhh.orig when hybrid fallback is not enabled
    if need_dbt:
        orig_blob_bytes = analysis.raw_bytes
    else:
        orig_blob_bytes = b""

    meta_va = _align_up(analysis.max_vaddr, 0x1000)
    meta_off = _align_up(len(analysis.raw_bytes), 0x1000)

    cur_meta_off = 0
    phdr_off_in_meta = cur_meta_off
    cur_meta_off = _align_up(cur_meta_off + phdr_table_sz, 16)

    interp_off_in_meta = cur_meta_off
    cur_meta_off = _align_up(cur_meta_off + len(interp_bytes), 16)

    hash_off_in_meta = cur_meta_off
    cur_meta_off = _align_up(cur_meta_off + len(hash_bytes), 16)

    dynsym_off_in_meta = cur_meta_off
    cur_meta_off = _align_up(cur_meta_off + dynsym_size, 16)

    dynstr_off_in_meta = cur_meta_off
    cur_meta_off = _align_up(cur_meta_off + len(dynstr_bytes), 16)

    rela_dyn_off_in_meta = cur_meta_off
    cur_meta_off = _align_up(cur_meta_off + rela_dyn_sz, 16)

    rela_plt_off_in_meta = cur_meta_off
    cur_meta_off = _align_up(cur_meta_off + rela_plt_sz, 16)

    if has_pinned_ver:
        versym_off_in_meta = cur_meta_off
        cur_meta_off = _align_up(cur_meta_off + len(versym_bytes), 16)
        verneed_off_in_meta = cur_meta_off
        cur_meta_off = _align_up(cur_meta_off + len(verneed_bytes), 16)
    else:
        versym_off_in_meta = 0
        verneed_off_in_meta = 0

    note_off_in_meta = cur_meta_off
    cur_meta_off = _align_up(cur_meta_off + len(note_bytes), 16)

    orig_blob_off_in_meta = cur_meta_off
    if len(orig_blob_bytes) > 0:
        cur_meta_off = _align_up(cur_meta_off + len(orig_blob_bytes), 16)

    meta_sz = cur_meta_off

    # 7. Run rewriter at exact rx_va!
    rx_va = _align_up(meta_va + meta_sz, 0x1000)
    rx_off = _align_up(meta_off + meta_sz, 0x1000)

    dispatch_vec_sz = _align_up((3 + len(initial_rewritten.plt_entries)) * 8, 16)
    rewritten_base_va = rx_va + dispatch_vec_sz
    rewritten = rewrite_binary(
        analysis,
        hybrid_mode=hybrid_mode,
        verbose=verbose,
        rewritten_base_va=rewritten_base_va,
    )
    rewritten_len_aligned = _align_up(len(rewritten.code_bytes), 16)

    # 8. Prepare trampoline targets & symbols
    x32_entry_targets: List[Tuple[str, int]] = []
    init_array_labels: List[str] = []
    fini_array_labels: List[str] = []
    export_targets: List[Tuple[str, int]] = []
    export_labels: Dict[str, str] = {}

    def _cdecl_va(orig_va: int) -> int:
        if orig_va in rewritten.orig_to_new:
            return rewritten.orig_to_new[orig_va]
        return orig_va

    for idx, iva in enumerate(analysis.init_array):
        lbl = f"__nm_initarr_{idx}"
        x32_entry_targets.append((lbl, _cdecl_va(iva)))
        init_array_labels.append(lbl)

    for idx, fva in enumerate(analysis.fini_array):
        lbl = f"__nm_finiarr_{idx}"
        x32_entry_targets.append((lbl, _cdecl_va(fva)))
        fini_array_labels.append(lbl)

    for sname, sva, _didx in exported_funcs:
        lbl = f"__nm_exp_{sname}"
        export_targets.append((lbl, _cdecl_va(sva)))
        export_labels[sname] = lbl

    # Dual-ABI stubs for address-taken internal functions
    dual_abi_targets: List[Tuple[str, int]] = []
    dual_va_set = set(analysis.addr_taken_funcs)
    if analysis.e_entry:
        dual_va_set.add(analysis.e_entry)
    for fva in sorted(dual_va_set):
        if fva in rewritten.orig_to_new:
            lbl = f"__nm_dual_{fva:08x}"
            dual_abi_targets.append((lbl, rewritten.orig_to_new[fva]))

    # orig_to_new pairs for runtime lookup
    orig_to_new_pairs: List[Tuple[int, int]] = []
    for fva in sorted(analysis.functions):
        if fva in rewritten.orig_to_new:
            orig_to_new_pairs.append((fva, rewritten.orig_to_new[fva]))

    # R_386_COPY slots for stdio and environ/getopt
    copy_stdio_slots: List[Tuple[str, int]] = []
    copy_data_slots: Dict[str, int] = {}
    for r in analysis.rel_dyn:
        if r.r_type == 5:
            if r.sym_name in ("stdin", "stdout", "stderr"):
                copy_stdio_slots.append((r.sym_name, r.offset))
            if r.sym_name in ("environ", "__environ", "_environ"):
                copy_data_slots["environ"] = r.offset
            elif r.sym_name in ("optarg", "optind", "opterr", "optopt"):
                copy_data_slots[r.sym_name] = r.offset

    # R_386_IRELATIVE resolver thunks (__nm_ifunc_<i>)
    ifunc_thunks: List[Tuple[int, int]] = []
    ifunc_relocs: Dict[int, int] = {}
    for r in analysis.rel_dyn + analysis.rel_plt:
        if r.r_type == 42:  # R_386_IRELATIVE
            orig_resolver_va = _read_u32_at_va(analysis.phdrs, analysis.raw_bytes, r.offset)
            thunk_idx = len(ifunc_thunks)
            ifunc_thunks.append((thunk_idx, orig_resolver_va))
            ifunc_relocs[r.offset] = thunk_idx

    # Resolve entries for all PLT stubs
    plt_entries_resolved: List[LookupSymEntry] = []
    for orig_sym, _gva, _sidx in rewritten.plt_entries:
        if orig_sym.startswith("__nm_ifunc_"):
            ent = LookupSymEntry(
                soname="",
                sym=orig_sym,
                tgt_soname="",
                tgt_sym=orig_sym,
                tgt_ver=None,
                cls="PATCHED_CDECL",
                sig_str="",
                parsed_sig=None,
                flags=0,
                raw_flags="",
                size=0,
            )
            resolved_imports[orig_sym] = ent
        else:
            ent = resolved_imports.get(orig_sym)
            if ent is None:
                orig_so, orig_v = analysis.sym_version_map.get(orig_sym, (None, None))
                ent = resolve_import_binding(
                    lookup_table,
                    orig_sym,
                    orig_so,
                    orig_v,
                    allow_untyped_unsafe=allow_untyped_unsafe,
                    dt_needed=analysis.dt_needed,
                )
        plt_entries_resolved.append(ent)

    # Level >= 2: specialized stubs via stubgen
    plt_stub_asm_override = None
    if level >= 2:
        from .stubgen import generate_specialized_stub, classify_stub

        override_dict: Dict[int, str] = {}
        for idx, (orig_sym, _gva, _sidx) in enumerate(rewritten.plt_entries):
            ent = plt_entries_resolved[idx]
            cls = classify_stub(ent)
            if cls in ("PATCHED_CDECL", "CDECL_DIRECT", "NATIVE_TYPED"):
                stub_lines = generate_specialized_stub(idx, orig_sym, ent)
                override_dict[idx] = "\n".join(stub_lines)
        plt_stub_asm_override = override_dict

    # Build helper assembly
    raw_asm = generate_helper_assembly(
        plt_entries=rewritten.plt_entries,
        x32_entry_targets=x32_entry_targets,
        dual_abi_targets=dual_abi_targets,
        fallback_funcs=rewritten.fallback_funcs,
        orig_to_new_pairs=orig_to_new_pairs,
        copy_stdio_slots=copy_stdio_slots,
        tls_info=analysis.tls_info,
        orig_init_va=analysis.init_va,
        orig_entry_va=analysis.e_entry,
        is_executable=analysis.has_interp,
        e_type=analysis.e_type,
        use_dbt_embedded_entry=use_dbt_embedded_entry,
        orig_elf_va=meta_va + orig_blob_off_in_meta if len(orig_blob_bytes) > 0 else 0,
        orig_elf_size=len(orig_blob_bytes),
        ref_va_for_bias=rx_va,
        init_array_labels=init_array_labels,
        fini_array_labels=fini_array_labels,
        lookup_table=lookup_table,
        plt_entries_resolved=plt_entries_resolved,
        copy_data_slots=copy_data_slots,
        ifunc_thunks=ifunc_thunks,
        plt_stub_asm_override=plt_stub_asm_override,
        export_targets=export_targets,
    )

    extra_objs: List[Tuple[str, bytes]] = []
    if boost_report is not None and getattr(boost_report, "boosted_map", None):
        for b_va, bf in sorted(boost_report.boosted_map.items()):
            extra_objs.append((f"l3_{b_va:x}", bf.obj_bytes))
            raw_asm += "\n" + bf.boundary_stub_asm + "\n"

    marker = "__nm_ensure_gs_init:"
    gap_asm = f"    .p2align 4\n__nm_rewritten_gap:\n    .skip {rewritten_len_aligned}\n\n"
    if marker not in raw_asm:
        raise PatcherError("internal error: __nm_ensure_gs_init marker missing")
    full_asm = raw_asm.replace(marker, gap_asm + marker, 1)

    # Compute exact .dynamic size dynamically
    # Count tags: needed_libs + soname + init + fini + arrays + hash + strtab + symtab
    # + strsz + syment + pltgot + rela + relasz + relaent + jmprel + flags + flags1 + bind_now + null
    # + versym/verneed if pinned
    num_dyn_tags = len(needed_str_offs) + 15
    if soname_dyn_off:
        num_dyn_tags += 1
    has_init_arr = bool((analysis.dyn_tags.get(DT_INIT_ARRAY, 0) and analysis.dyn_tags.get(DT_INIT_ARRAYSZ, 0)) or init_array_labels)
    if has_init_arr:
        num_dyn_tags += 2
    has_fini_arr = bool((analysis.dyn_tags.get(DT_FINI_ARRAY, 0) and analysis.dyn_tags.get(DT_FINI_ARRAYSZ, 0)) or fini_array_labels)
    if has_fini_arr:
        num_dyn_tags += 2
    if analysis.rel_plt:
        num_dyn_tags += 3
    if has_pinned_ver:
        num_dyn_tags += 3  # DT_VERSYM, DT_VERNEED, DT_VERNEEDNUM
    dynamic_reserve = _align_up((num_dyn_tags + 4) * 8, 16)

    def _make_defsyms(temp_rw_va: int, rx_end_est: int) -> Dict[str, int]:
        ds: Dict[str, int] = {
            "sym_text_new_start": rx_va,
            "sym_text_new_end": rx_end_est,
            "sym_orig_elf_blob": meta_va + orig_blob_off_in_meta if len(orig_blob_bytes) > 0 else 0,
        }
        if analysis.tls_info is not None:
            ds["sym_orig_tls_data"] = analysis.tls_info[0]
        if analysis.init_va:
            ds["sym_rewritten_init"] = _cdecl_va(analysis.init_va)
        if analysis.e_entry:
            ds["sym_rewritten_entry"] = _cdecl_va(analysis.e_entry)
        for lbl, nva in x32_entry_targets:
            ds[f"target_{lbl}"] = nva
        for lbl, nva in dual_abi_targets:
            ds[f"target_{lbl}"] = nva
        for lbl, nva in export_targets:
            ds[f"target_{lbl}"] = nva
        for thunk_idx, orig_res in ifunc_thunks:
            ds[f"target___nm_ifunc_{thunk_idx}"] = _cdecl_va(orig_res)
        for idx, (orig_sym, got_slot_va, _slot_idx) in enumerate(rewritten.plt_entries):
            if got_slot_va != 0:
                slot_va = got_slot_va
            else:
                if idx not in extra_plt_got_slots:
                    raise PatcherError(f"PLT entry {idx} ({orig_sym}) has got_slot_va == 0 but no extra GOT slot allocated")
                g_idx, _ = extra_plt_got_slots[idx]
                slot_va = (temp_rw_va + dynamic_reserve) + g_idx * 4
            if slot_va == 0:
                raise PatcherError(f"sym_got_plt_{idx} cannot be 0!")
            ds[f"sym_got_plt_{idx}"] = slot_va
        for sname, dst_va in copy_stdio_slots:
            ds[f"sym_copy_{sname}"] = dst_va
        for sname, dst_va in copy_data_slots.items():
            ds[f"sym_copy_{sname}"] = dst_va

        got_base = temp_rw_va + dynamic_reserve
        for g_idx, gname in enumerate(extra_got_syms):
            ds[f"got_{gname}"] = got_base + g_idx * 4
        return ds

    # Pass A: measure sizes
    t_bytes_a, r_bytes_a, _d_a, _bss_sz_a, sym_vas_a = assemble_and_link_helpers(
        full_asm,
        text_va=rx_va,
        rodata_va=rx_va + 0x400000,
        data_va=rx_va + 0x800000,
        bss_va=rx_va + 0xC00000,
        defsyms=_make_defsyms(rx_va + 0x800000, rx_va + 0x100000),
        extra_objs=extra_objs if extra_objs else None,
    )

    rx_sz = _align_up(len(t_bytes_a), 16)
    rodata_va = _align_up(rx_va + rx_sz, 0x1000)
    rodata_off = _align_up(rx_off + rx_sz, 0x1000)

    helpers_ro_sz = _align_up(len(r_bytes_a), 16)
    jt_base_va = rodata_va + helpers_ro_sz
    rodata_total_sz = helpers_ro_sz + _align_up(rewritten.rodata_jump_tables_size, 16)

    rw_va = _align_up(rodata_va + rodata_total_sz, 0x1000)
    rw_off = _align_up(rodata_off + rodata_total_sz, 0x1000)

    extra_got_base_va = rw_va + dynamic_reserve
    extra_got_sz = len(extra_got_syms) * 4
    bss_va = _align_up(extra_got_base_va + extra_got_sz, 16)

    # Pass B: assemble & link at exact VAs
    defsyms_final = _make_defsyms(rw_va, rx_va + rx_sz)
    for g_idx, gname in enumerate(extra_got_syms):
        defsyms_final[f"got_{gname}"] = extra_got_base_va + g_idx * 4
    if not need_dbt:
        defsyms_final["got_newminhh_dbt_enter_at"] = 0
        defsyms_final["got_newminhh_dbt_exec_embedded"] = 0

    if boost_report is not None and getattr(boost_report, "boosted_map", None):
        for b_va in boost_report.boosted_map:
            entry_sym = f"__nm_l3_entry_{b_va:x}"
            if entry_sym in sym_vas_a:
                l3_nva = sym_vas_a[entry_sym]
                for sname, sva, _ in exported_funcs:
                    if sva == b_va:
                        defsyms_final[f"target___nm_exp_{sname}"] = l3_nva
                if b_va in dual_va_set:
                    defsyms_final[f"target___nm_dual_{b_va:08x}"] = l3_nva

    t_bytes, r_bytes, _d_bytes, bss_sz, sym_vas = assemble_and_link_helpers(
        full_asm,
        text_va=rx_va,
        rodata_va=rodata_va,
        data_va=extra_got_base_va,
        bss_va=bss_va,
        defsyms=defsyms_final,
        extra_objs=extra_objs if extra_objs else None,
    )

    if len(t_bytes) > rx_sz:
        raise PatcherError(f"helper .text grew between pass A ({rx_sz}) and pass B ({len(t_bytes)})")

    # Splice rewritten.code_bytes into gap
    rx_buf = bytearray(t_bytes)
    gap_start = dispatch_vec_sz
    rx_buf[gap_start : gap_start + len(rewritten.code_bytes)] = rewritten.code_bytes

    if boost_report is not None and getattr(boost_report, "boosted_map", None):
        for b_va in boost_report.boosted_map:
            entry_sym = f"__nm_l3_entry_{b_va:x}"
            if entry_sym in sym_vas:
                l3_entry_va = sym_vas[entry_sym]
                old_nva = rewritten.orig_to_new.get(b_va)
                rewritten.orig_to_new[b_va] = l3_entry_va
                if old_nva is not None:
                    code_off = old_nva - (rx_va + dispatch_vec_sz)
                    if 0 <= code_off <= len(rewritten.code_bytes) - 5:
                        rel32 = l3_entry_va - (old_nva + 5)
                        struct.pack_into("<Bi", rx_buf, gap_start + code_off, 0xE9, rel32)

    for patch_off, next_insn_va, jt_off in rewritten.rodata_fixups:
        target_va = jt_base_va + jt_off
        struct.pack_into("<i", rx_buf, gap_start + patch_off, target_va - next_insn_va)

    for patch_off, next_insn_va, lbl in rewritten.label_fixups:
        if lbl not in sym_vas:
            raise PatcherError(f"unresolved helper label '{lbl}' during ELF link")
        target_va = sym_vas[lbl]
        struct.pack_into("<i", rx_buf, gap_start + patch_off, target_va - next_insn_va)

    # 9. Build .rodata.newminhh
    rodata_buf = bytearray(r_bytes.ljust(helpers_ro_sz, b"\x00"))
    for jt_blob in rewritten.jump_table_blobs:
        jt_va = jt_base_va + jt_blob.rodata_offset
        for orig_t in jt_blob.targets:
            new_t = rewritten.orig_to_new[orig_t]
            rodata_buf.extend(struct.pack("<i", new_t - jt_va))
    while len(rodata_buf) < rodata_total_sz:
        rodata_buf.append(0)

    # 10. Patch original image bytes
    orig_patched = bytearray(analysis.raw_bytes)
    struct.pack_into("<H", orig_patched, 18, EM_X86_64)

    new_entry = sym_vas["_nm_start"] if analysis.has_interp else 0
    struct.pack_into("<I", orig_patched, 24, new_entry)

    init_arr_va = analysis.dyn_tags.get(DT_INIT_ARRAY, 0)
    init_arr_sz = analysis.dyn_tags.get(DT_INIT_ARRAYSZ, 0)
    fini_arr_va = analysis.dyn_tags.get(DT_FINI_ARRAY, 0)
    fini_arr_sz = analysis.dyn_tags.get(DT_FINI_ARRAYSZ, 0)

    if init_arr_va == 0 and init_array_labels:
        init_arr_va = sym_vas.get("__nm_synthetic_init_array", 0)
        init_arr_sz = len(init_array_labels) * 4

    if fini_arr_va == 0 and fini_array_labels:
        fini_arr_va = sym_vas.get("__nm_synthetic_fini_array", 0)
        fini_arr_sz = len(fini_array_labels) * 4

    init_arr_slot_map: Dict[int, int] = {}
    if init_arr_va:
        for i, lbl in enumerate(init_array_labels):
            init_arr_slot_map[init_arr_va + i * 4] = sym_vas[lbl]

    fini_arr_slot_map: Dict[int, int] = {}
    if fini_arr_va:
        for i, lbl in enumerate(fini_array_labels):
            fini_arr_slot_map[fini_arr_va + i * 4] = sym_vas[lbl]

    if analysis.e_type == 2:  # ET_EXEC: overwrite .init_array / .fini_array slots directly
        for slot_va, shim_va in list(init_arr_slot_map.items()) + list(fini_arr_slot_map.items()):
            if slot_va != 0:
                foff = _vaddr_to_offset(analysis.phdrs, slot_va)
                if foff is not None:
                    struct.pack_into("<I", orig_patched, foff, shim_va)

    # Update .dynsym exports
    for sname, lbl in export_labels.items():
        didx = sym_name_to_idx[sname]
        new_dynsyms[didx].st_value = sym_vas[lbl]

    # Update __nm_cdecl.<sym> exports
    if analysis.e_type == 3:  # ET_DYN
        for sname, sva, _didx in exported_funcs:
            cdecl_alias_name = f"__nm_cdecl.{sname}"
            if cdecl_alias_name in sym_name_to_idx:
                cidx = sym_name_to_idx[cdecl_alias_name]
                new_dynsyms[cidx].st_value = _cdecl_va(sva)

    # Map symbol names to canonical PLT stub VA
    sym_to_plt_idx: Dict[str, int] = {}
    for idx, (pname, _gva, _sidx) in enumerate(rewritten.plt_entries):
        sym_to_plt_idx[pname] = idx

    # 11. Build .rela.dyn and .rela.plt
    rela_dyn_bytes = bytearray()
    for r in analysis.rel_dyn:
        if r.r_type == 8:  # R_386_RELATIVE -> R_X86_64_RELATIVE
            orig_val = _read_u32_at_va(analysis.phdrs, analysis.raw_bytes, r.offset)
            if r.offset in init_arr_slot_map:
                addend = init_arr_slot_map[r.offset]
            elif r.offset in fini_arr_slot_map:
                addend = fini_arr_slot_map[r.offset]
            elif f"__nm_dual_{orig_val:08x}" in sym_vas:
                addend = sym_vas[f"__nm_dual_{orig_val:08x}"]
            elif orig_val in rewritten.orig_to_new:
                addend = rewritten.orig_to_new[orig_val]
            else:
                addend = orig_val
            rela_dyn_bytes.extend(struct.pack("<IIi", r.offset, R_X86_64_RELATIVE, addend))
        elif r.r_type == 6:  # R_386_GLOB_DAT -> route address-taken imports to canonical PLT stub
            orig_sym = analysis.dynsyms[r.sym_idx] if r.sym_idx < len(analysis.dynsyms) else None
            is_global_func_import = (
                orig_sym is not None
                and orig_sym.st_shndx == 0
                and orig_sym.sym_type == 2
                and orig_sym.bind == 1
            )
            if r.sym_name in sym_to_plt_idx and is_global_func_import:
                p_idx = sym_to_plt_idx[r.sym_name]
                p_va = sym_vas[f"__nm_plt_{p_idx}"]
                rela_dyn_bytes.extend(struct.pack("<IIi", r.offset, R_X86_64_RELATIVE, p_va))
            else:
                rela_dyn_bytes.extend(
                    struct.pack("<IIi", r.offset, (r.sym_idx << 8) | R_X86_64_GLOB_DAT, 0)
                )
        elif r.r_type == 1:  # R_386_32 -> route address-taken imports to canonical PLT stub
            orig_sym = analysis.dynsyms[r.sym_idx] if r.sym_idx < len(analysis.dynsyms) else None
            is_global_func_import = (
                orig_sym is not None
                and orig_sym.st_shndx == 0
                and orig_sym.sym_type == 2
                and orig_sym.bind == 1
            )
            if r.sym_name in sym_to_plt_idx and is_global_func_import:
                p_idx = sym_to_plt_idx[r.sym_name]
                p_va = sym_vas[f"__nm_plt_{p_idx}"]
                rela_dyn_bytes.extend(struct.pack("<IIi", r.offset, R_X86_64_RELATIVE, p_va))
            else:
                orig_val = _read_u32_at_va(analysis.phdrs, analysis.raw_bytes, r.offset)
                rela_dyn_bytes.extend(
                    struct.pack("<IIi", r.offset, (r.sym_idx << 8) | R_X86_64_32, orig_val)
                )
        elif r.r_type == 5:  # R_386_COPY -> R_X86_64_COPY
            rela_dyn_bytes.extend(
                struct.pack("<IIi", r.offset, (r.sym_idx << 8) | R_X86_64_COPY, 0)
            )
        elif r.r_type == 2:  # R_386_PC32 -> R_X86_64_PC32
            orig_val = _read_u32_at_va(analysis.phdrs, analysis.raw_bytes, r.offset)
            rela_dyn_bytes.extend(
                struct.pack("<IIi", r.offset, (r.sym_idx << 8) | R_X86_64_PC32, orig_val)
            )
        elif r.r_type == 42:  # R_386_IRELATIVE -> R_X86_64_IRELATIVE
            thunk_idx = ifunc_relocs[r.offset]
            ifunc_shim_va = sym_vas[f"__nm_ifunc_{thunk_idx}"]
            rela_dyn_bytes.extend(struct.pack("<IIi", r.offset, R_X86_64_IRELATIVE, ifunc_shim_va))
        elif r.r_type == 14 or r.r_type == 37:  # R_386_TLS_TPOFF / TPOFF32 -> R_X86_64_TPOFF32
            orig_val = _read_u32_at_va(analysis.phdrs, analysis.raw_bytes, r.offset)
            rela_dyn_bytes.extend(
                struct.pack("<IIi", r.offset, (r.sym_idx << 8) | R_X86_64_TPOFF32, orig_val)
            )
        elif r.r_type == 35:  # R_386_TLS_DTPMOD32 -> R_X86_64_DTPMOD64
            rela_dyn_bytes.extend(
                struct.pack("<IIi", r.offset, (r.sym_idx << 8) | R_X86_64_DTPMOD64, 0)
            )
        elif r.r_type == 36:  # R_386_TLS_DTPOFF32 -> R_X86_64_DTPOFF32
            orig_val = _read_u32_at_va(analysis.phdrs, analysis.raw_bytes, r.offset)
            rela_dyn_bytes.extend(
                struct.pack("<IIi", r.offset, (r.sym_idx << 8) | R_X86_64_DTPOFF32, orig_val)
            )
        else:
            raise PatcherError(f"unsupported i386 relocation type {r.r_type} at 0x{r.offset:08x}")

    for g_idx, gname in enumerate(extra_got_syms):
        slot_va = extra_got_base_va + g_idx * 4
        sidx = sym_name_to_idx[gname]
        rela_dyn_bytes.extend(struct.pack("<IIi", slot_va, (sidx << 8) | R_X86_64_GLOB_DAT, 0))

    rela_plt_bytes = bytearray()
    for r in analysis.rel_plt:
        if r.r_type == 42:  # R_386_IRELATIVE
            thunk_idx = ifunc_relocs[r.offset]
            ifunc_shim_va = sym_vas[f"__nm_ifunc_{thunk_idx}"]
            rela_plt_bytes.extend(struct.pack("<IIi", r.offset, R_X86_64_IRELATIVE, ifunc_shim_va))
        else:
            rela_plt_bytes.extend(
                struct.pack("<IIi", r.offset, (r.sym_idx << 8) | R_X86_64_JUMP_SLOT, 0)
            )

    # 12. Assemble ext_ro_meta segment bytes
    hash_va = meta_va + hash_off_in_meta
    dynsym_va = meta_va + dynsym_off_in_meta
    dynstr_va = meta_va + dynstr_off_in_meta
    rela_dyn_va = meta_va + rela_dyn_off_in_meta
    rela_plt_va = meta_va + rela_plt_off_in_meta
    note_va = meta_va + note_off_in_meta
    interp_va = meta_va + interp_off_in_meta
    versym_va = meta_va + versym_off_in_meta if has_pinned_ver else 0
    verneed_va = meta_va + verneed_off_in_meta if has_pinned_ver else 0

    # Build .dynamic in ext_rw (at rw_va)
    dyn_entries: List[Tuple[int, int]] = []
    for noff in needed_str_offs:
        dyn_entries.append((DT_NEEDED, noff))
    if soname_dyn_off:
        dyn_entries.append((DT_SONAME, soname_dyn_off))
    dyn_entries.append((DT_INIT, sym_vas["_nm_init"]))
    dyn_entries.append((DT_FINI, sym_vas["_nm_fini"]))
    if has_init_arr and init_arr_va and init_arr_sz:
        dyn_entries.append((DT_INIT_ARRAY, init_arr_va))
        dyn_entries.append((DT_INIT_ARRAYSZ, init_arr_sz))
    if has_fini_arr and fini_arr_va and fini_arr_sz:
        dyn_entries.append((DT_FINI_ARRAY, fini_arr_va))
        dyn_entries.append((DT_FINI_ARRAYSZ, fini_arr_sz))
    dyn_entries.append((DT_HASH, hash_va))
    dyn_entries.append((DT_STRTAB, dynstr_va))
    dyn_entries.append((DT_SYMTAB, dynsym_va))
    dyn_entries.append((DT_STRSZ, len(dynstr_bytes)))
    dyn_entries.append((DT_SYMENT, 16))
    if analysis.got_plt_va:
        dyn_entries.append((DT_PLTGOT, analysis.got_plt_va))
    dyn_entries.append((DT_RELA, rela_dyn_va))
    dyn_entries.append((DT_RELASZ, len(rela_dyn_bytes)))
    dyn_entries.append((DT_RELAENT, 12))
    if rela_plt_bytes:
        dyn_entries.append((DT_JMPREL, rela_plt_va))
        dyn_entries.append((DT_PLTRELSZ, len(rela_plt_bytes)))
        dyn_entries.append((DT_PLTREL, DT_RELA))
    if has_pinned_ver:
        dyn_entries.append((DT_VERSYM, versym_va))
        dyn_entries.append((DT_VERNEED, verneed_va))
        dyn_entries.append((DT_VERNEEDNUM, num_verneed))
    dyn_entries.append((DT_FLAGS, 0x8))  # DF_BIND_NOW
    flags1 = 0x1  # DF_1_NOW
    if analysis.e_type == 3 and analysis.has_interp:
        flags1 |= 0x08000000  # DF_1_PIE
    dyn_entries.append((DT_FLAGS_1, flags1))
    dyn_entries.append((DT_BIND_NOW, 0))
    dyn_entries.append((DT_NULL, 0))

    dynamic_bytes = bytearray()
    for tag, val in dyn_entries:
        dynamic_bytes.extend(struct.pack("<iI", tag, val))
    if len(dynamic_bytes) > dynamic_reserve:
        raise PatcherError(f"internal error: .dynamic ({len(dynamic_bytes)}) exceeded reserve ({dynamic_reserve})")

    ext_rw_filesz = dynamic_reserve + extra_got_sz
    ext_rw_memsz = (bss_va - rw_va) + bss_sz
    ext_rw_buf = bytearray(ext_rw_filesz)
    ext_rw_buf[0 : len(dynamic_bytes)] = dynamic_bytes

    # 13. Build Program Header Table
    final_phdrs: List[PhdrInfo] = []
    orig_loads = [p for p in kept_orig_phdrs if p.p_type == PT_LOAD]
    other_phdrs = [p for p in kept_orig_phdrs if p.p_type != PT_LOAD]

    new_loads = list(orig_loads) + [
        PhdrInfo(PT_LOAD, meta_off, meta_va, meta_va, meta_sz, meta_sz, PF_R, 0x1000),
        PhdrInfo(PT_LOAD, rx_off, rx_va, rx_va, len(rx_buf), len(rx_buf), PF_R | PF_X, 0x1000),
        PhdrInfo(PT_LOAD, rodata_off, rodata_va, rodata_va, len(rodata_buf), len(rodata_buf), PF_R, 0x1000),
        PhdrInfo(PT_LOAD, rw_off, rw_va, rw_va, ext_rw_filesz, ext_rw_memsz, PF_R | PF_W, 0x1000),
    ]
    new_loads.sort(key=lambda p: p.p_vaddr)

    for p in other_phdrs:
        if p.p_type == PT_PHDR:
            final_phdrs.append(
                PhdrInfo(PT_PHDR, meta_off, meta_va, meta_va, new_phnum * 32, new_phnum * 32, PF_R, 4)
            )
        elif p.p_type == PT_INTERP and analysis.has_interp:
            final_phdrs.append(
                PhdrInfo(PT_INTERP, meta_off + interp_off_in_meta, interp_va, interp_va, len(interp_bytes), len(interp_bytes), PF_R, 1)
            )
        elif p.p_type == PT_DYNAMIC:
            final_phdrs.append(
                PhdrInfo(PT_DYNAMIC, rw_off, rw_va, rw_va, len(dynamic_bytes), len(dynamic_bytes), PF_R | PF_W, 4)
            )
        elif p.p_type == 0x6474E551:  # PT_GNU_STACK: always strip PF_X to enforce NX stack
            final_phdrs.append(
                PhdrInfo(p.p_type, p.p_offset, p.p_vaddr, p.p_paddr, p.p_filesz, p.p_memsz, p.p_flags & ~PF_X, p.p_align)
            )
        else:
            final_phdrs.append(p)

    head_phdrs = [p for p in final_phdrs if p.p_type in (PT_PHDR, PT_INTERP)]
    tail_phdrs = [p for p in final_phdrs if p.p_type not in (PT_PHDR, PT_INTERP)]
    note_newminhh_phdr = PhdrInfo(
        PT_NOTE, meta_off + note_off_in_meta, note_va, note_va, len(note_bytes), len(note_bytes), PF_R, 4
    )
    ordered_phdrs = head_phdrs + new_loads + tail_phdrs + [note_newminhh_phdr]
    if len(ordered_phdrs) != new_phnum:
        raise PatcherError(f"phdr count mismatch: {len(ordered_phdrs)} != {new_phnum}")

    phdr_bytes = bytearray()
    for p in ordered_phdrs:
        phdr_bytes.extend(
            struct.pack(
                "<IIIIIIII",
                p.p_type,
                p.p_offset,
                p.p_vaddr,
                p.p_paddr,
                p.p_filesz,
                p.p_memsz,
                p.p_flags,
                p.p_align,
            )
        )

    # 14. Assemble ext_ro_meta buffer
    dynsym_bytes = bytearray()
    for s in new_dynsyms:
        name_off = str_offsets[s.name]
        dynsym_bytes.extend(
            struct.pack(
                "<IIIBBH",
                name_off,
                s.st_value,
                s.st_size,
                s.st_info,
                s.st_other,
                s.st_shndx,
            )
        )

    meta_buf = bytearray(meta_sz)
    meta_buf[phdr_off_in_meta : phdr_off_in_meta + len(phdr_bytes)] = phdr_bytes
    if interp_bytes:
        meta_buf[interp_off_in_meta : interp_off_in_meta + len(interp_bytes)] = interp_bytes
    meta_buf[hash_off_in_meta : hash_off_in_meta + len(hash_bytes)] = hash_bytes
    meta_buf[dynsym_off_in_meta : dynsym_off_in_meta + len(dynsym_bytes)] = dynsym_bytes
    meta_buf[dynstr_off_in_meta : dynstr_off_in_meta + len(dynstr_bytes)] = dynstr_bytes
    meta_buf[rela_dyn_off_in_meta : rela_dyn_off_in_meta + len(rela_dyn_bytes)] = rela_dyn_bytes
    if rela_plt_bytes:
        meta_buf[rela_plt_off_in_meta : rela_plt_off_in_meta + len(rela_plt_bytes)] = rela_plt_bytes
    if has_pinned_ver:
        meta_buf[versym_off_in_meta : versym_off_in_meta + len(versym_bytes)] = versym_bytes
        meta_buf[verneed_off_in_meta : verneed_off_in_meta + len(verneed_bytes)] = verneed_bytes
    meta_buf[note_off_in_meta : note_off_in_meta + len(note_bytes)] = note_bytes
    if len(orig_blob_bytes) > 0:
        meta_buf[orig_blob_off_in_meta : orig_blob_off_in_meta + len(orig_blob_bytes)] = orig_blob_bytes

    # 15. Build output buffer
    out_buf = bytearray(orig_patched)
    out_buf.extend(b"\x00" * (meta_off - len(out_buf)))
    out_buf.extend(meta_buf)
    out_buf.extend(b"\x00" * (rx_off - len(out_buf)))
    out_buf.extend(rx_buf)
    out_buf.extend(b"\x00" * (rodata_off - len(out_buf)))
    out_buf.extend(rodata_buf)
    out_buf.extend(b"\x00" * (rw_off - len(out_buf)))
    out_buf.extend(ext_rw_buf)

    # 16. Section Header Table
    e_shstrndx = struct.unpack_from("<H", analysis.raw_bytes, 50)[0]
    if analysis.shdrs and e_shstrndx < len(analysis.shdrs):
        s = analysis.shdrs[e_shstrndx]
        shstr_sec = bytearray(analysis.raw_bytes[s.sh_offset : s.sh_offset + s.sh_size])
    else:
        shstr_sec = bytearray(b"\x00")

    def _add_shstr(name: str) -> int:
        b = name.encode("ascii") + b"\x00"
        pos = shstr_sec.find(b"\x00" + b)
        if pos >= 0:
            return pos + 1
        off = len(shstr_sec)
        shstr_sec.extend(b)
        return off

    new_shdrs: List[Tuple[int, int, int, int, int, int, int, int, int, int]] = []
    if analysis.shdrs:
        for s in analysis.shdrs:
            name_off = struct.unpack_from(
                "<I",
                analysis.raw_bytes,
                struct.unpack_from("<I", analysis.raw_bytes, 32)[0] + s.index * 40,
            )[0]
            if s.name in (
                ".dynsym", ".dynstr", ".hash", ".gnu.hash", ".gnu.version", ".gnu.version_r",
                ".rel.dyn", ".rel.plt", ".dynamic", ".interp"
            ):
                new_name_off = _add_shstr(".orig" + s.name)
                new_shdrs.append(
                    (new_name_off, 1, s.sh_flags, s.sh_addr, s.sh_offset, s.sh_size, 0, 0, s.sh_addralign, 0)
                )
            elif s.index == e_shstrndx:
                new_shdrs.append((name_off, s.sh_type, s.sh_flags, s.sh_addr, s.sh_offset, s.sh_size, s.sh_link, s.sh_info, s.sh_addralign, s.sh_entsize))
            else:
                new_shdrs.append(
                    (name_off, s.sh_type, s.sh_flags, s.sh_addr, s.sh_offset, s.sh_size, s.sh_link, s.sh_info, s.sh_addralign, s.sh_entsize)
                )
    else:
        new_shdrs.append((0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    if interp_bytes:
        new_shdrs.append((_add_shstr(".interp"), 1, 2, interp_va, meta_off + interp_off_in_meta, len(interp_bytes), 0, 0, 1, 0))
    hash_idx = len(new_shdrs)
    dynsym_idx = hash_idx + 1
    dynstr_idx = hash_idx + 2
    new_shdrs.append((_add_shstr(".hash"), 5, 2, hash_va, meta_off + hash_off_in_meta, len(hash_bytes), dynsym_idx, 0, 4, 4))
    new_shdrs.append((_add_shstr(".dynsym"), 11, 2, dynsym_va, meta_off + dynsym_off_in_meta, len(dynsym_bytes), dynstr_idx, 1, 4, 16))
    new_shdrs.append((_add_shstr(".dynstr"), 3, 2, dynstr_va, meta_off + dynstr_off_in_meta, len(dynstr_bytes), 0, 0, 1, 0))
    new_shdrs.append((_add_shstr(".rela.dyn"), 4, 2, rela_dyn_va, meta_off + rela_dyn_off_in_meta, len(rela_dyn_bytes), dynsym_idx, 0, 4, 12))
    if rela_plt_bytes:
        new_shdrs.append((_add_shstr(".rela.plt"), 4, 2, rela_plt_va, meta_off + rela_plt_off_in_meta, len(rela_plt_bytes), dynsym_idx, 0, 4, 12))
    if has_pinned_ver:
        new_shdrs.append((_add_shstr(".gnu.version"), SHT_GNU_versym, 2, versym_va, meta_off + versym_off_in_meta, len(versym_bytes), dynsym_idx, 0, 2, 2))
        new_shdrs.append((_add_shstr(".gnu.version_r"), SHT_GNU_verneed, 2, verneed_va, meta_off + verneed_off_in_meta, len(verneed_bytes), dynstr_idx, num_verneed, 4, 0))
    new_shdrs.append((_add_shstr(".rodata.newminhh"), 1, 2, rodata_va, rodata_off, len(rodata_buf), 0, 0, 16, 0))
    text_new_shndx = len(new_shdrs)
    new_shdrs.append((_add_shstr(".text.newminhh"), 1, 6, rx_va, rx_off, len(rx_buf), 0, 0, 16, 0))
    new_shdrs.append((_add_shstr(".dynamic"), 6, 3, rw_va, rw_off, len(dynamic_bytes), dynstr_idx, 0, 4, 8))
    new_shdrs.append((_add_shstr(".data.newminhh"), 1, 3, extra_got_base_va, rw_off + dynamic_reserve, extra_got_sz, 0, 0, 4, 0))
    new_shdrs.append((_add_shstr(".bss.newminhh"), 8, 3, bss_va, rw_off + ext_rw_filesz, bss_sz, 0, 0, 16, 0))
    new_shdrs.append((_add_shstr(".note.newminhh"), 7, 2, note_va, meta_off + note_off_in_meta, len(note_bytes), 0, 0, 4, 0))
    if len(orig_blob_bytes) > 0:
        new_shdrs.append((_add_shstr(".newminhh.orig"), 1, 2, meta_va + orig_blob_off_in_meta, meta_off + orig_blob_off_in_meta, len(orig_blob_bytes), 0, 0, 16, 0))

    # Re-pack dynsym with correct text_new_shndx for __nm_cdecl.<sym> exports
    if analysis.e_type == 3:  # ET_DYN
        dynsym_repack = bytearray()
        for s in new_dynsyms:
            name_off = str_offsets[s.name]
            shndx = text_new_shndx if s.name.startswith("__nm_cdecl.") else s.st_shndx
            dynsym_repack.extend(
                struct.pack(
                    "<IIIBBH",
                    name_off,
                    s.st_value,
                    s.st_size,
                    s.st_info,
                    s.st_other,
                    shndx,
                )
            )
        out_buf[meta_off + dynsym_off_in_meta : meta_off + dynsym_off_in_meta + len(dynsym_repack)] = dynsym_repack

    new_shstr_idx = len(new_shdrs)
    new_shstr_off = len(out_buf)
    out_buf.extend(shstr_sec)
    new_shdrs.append((_add_shstr(".shstrtab.newminhh"), 3, 0, 0, new_shstr_off, len(shstr_sec), 0, 0, 1, 0))
    out_buf[new_shstr_off:] = shstr_sec

    while len(out_buf) % 4 != 0:
        out_buf.append(0)
    new_shoff = len(out_buf)
    for sh in new_shdrs:
        out_buf.extend(struct.pack("<IIIIIIIIII", *sh))

    # Update ELF header fields: e_phoff, e_shoff, e_phnum, e_shnum, e_shstrndx
    struct.pack_into("<I", out_buf, 28, meta_off)
    struct.pack_into("<I", out_buf, 32, new_shoff)
    struct.pack_into("<H", out_buf, 44, new_phnum)
    struct.pack_into("<H", out_buf, 48, len(new_shdrs))
    struct.pack_into("<H", out_buf, 50, new_shstr_idx)

    return bytes(out_buf)
