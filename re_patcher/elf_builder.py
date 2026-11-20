"""Stage 2c — EM_X86_64 ELF32 (x32) Builder: DT_RELA, .dynsym/.hash, PT_INTERP, .text.newminhh, .note.newminhh."""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

from . import PatcherError
from .analyzer import AnalysisResult, DynSymInfo, PhdrInfo, RelocInfo
from .rewriter import RewrittenOutput, rewrite_binary
from .trampolines import (
    assemble_and_link_helpers,
    generate_helper_assembly,
    map_import_symbol_name,
)

EM_X86_64 = 62
X32_INTERP = b"/libx32/ld-linux-x32.so.2\x00"

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

# x86_64/x32 relocation types
R_X86_64_NONE = 0
R_X86_64_64 = 1
R_X86_64_PC32 = 2
R_X86_64_COPY = 5
R_X86_64_GLOB_DAT = 6
R_X86_64_JUMP_SLOT = 7
R_X86_64_RELATIVE = 8
R_X86_64_32 = 10

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


def _build_note_newminhh(hybrid_mode: str, num_funcs: int, num_fallback: int) -> bytes:
    name = b"NEWMINHH\x00"
    desc = struct.pack("<IIII", 1, {"static": 0, "auto": 1, "hybrid": 2}[hybrid_mode], num_funcs, num_fallback)
    name_pad = name + b"\x00" * ((4 - len(name) % 4) % 4)
    desc_pad = desc + b"\x00" * ((4 - len(desc) % 4) % 4)
    return struct.pack("<III", len(name), len(desc), 0x4E4D4831) + name_pad + desc_pad


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
) -> bytes:
    """Construct final EM_X86_64 ELF32 binary/SO with rewritten .text.newminhh and DT_RELA."""
    need_dbt = (hybrid_mode == "hybrid") or (len(initial_rewritten.fallback_funcs) > 0)
    fallback_va_set = {va for _lbl, va in initial_rewritten.fallback_funcs}
    use_dbt_embedded_entry = (
        (hybrid_mode == "hybrid")
        and (analysis.e_type == 3)
        and (analysis.e_entry != 0)
        and (analysis.e_entry in fallback_va_set)
    )

    # 1. Build dynsym list: clone original dynsyms (with compat_* renames) + extra runtime imports
    new_dynsyms: List[DynSymInfo] = []
    sym_name_to_idx: Dict[str, int] = {}
    exported_funcs: List[Tuple[str, int, int]] = []  # (sym_name, orig_va, dynsym_idx)

    for idx, sym in enumerate(analysis.dynsyms):
        mapped_name = sym.name
        if sym.st_shndx == 0 and sym.name:
            mapped_name = map_import_symbol_name(sym.name)
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

    extra_imports = [
        "newminhh_compat_syscall",
        "newminhh_call_x32_cdecl",
        "newminhh_invoke_cdecl_asm",
        "newminhh_get_stdio_streams",
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

    # 2. Build .dynstr and .hash
    needed_libs = ["libnewminhh_compat.so"]
    if need_dbt:
        needed_libs.append("libnewminhh_dbt.so")

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
        # Keep SONAME if original shared library had one
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
    dynstr_bytes = bytes(dynstr)

    # 3. Compute sizes for ext_ro_meta (Segment 1)
    # Kept original phdrs (drop PT_GNU_PROPERTY) + 4 new PT_LOADs + 1 PT_NOTE
    kept_orig_phdrs = [p for p in analysis.phdrs if p.p_type != PT_GNU_PROPERTY]
    # Ensure PT_PHDR exists if executable or already present; count new phdrs
    has_pt_phdr = any(p.p_type == PT_PHDR for p in kept_orig_phdrs)
    new_phnum = len(kept_orig_phdrs) + (0 if has_pt_phdr else 0) + 5  # 4 PT_LOADs + 1 PT_NOTE
    phdr_table_sz = new_phnum * 32

    interp_bytes = X32_INTERP if analysis.has_interp else b""
    rela_dyn_count = len(analysis.rel_dyn) + len(extra_got_syms)
    rela_dyn_sz = rela_dyn_count * 12
    rela_plt_sz = len(analysis.rel_plt) * 12
    note_bytes = _build_note_newminhh(
        hybrid_mode, len(analysis.functions), len(initial_rewritten.fallback_funcs)
    )
    orig_blob_bytes = analysis.raw_bytes

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

    note_off_in_meta = cur_meta_off
    cur_meta_off = _align_up(cur_meta_off + len(note_bytes), 16)

    orig_blob_off_in_meta = cur_meta_off
    cur_meta_off = _align_up(cur_meta_off + len(orig_blob_bytes), 16)

    meta_sz = cur_meta_off

    # 4. Run rewriter at exact rx_va!
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

    # 5. Prepare trampoline targets & symbols
    # x32->cdecl entry stubs: .dynsym exports, init_array, fini_array
    x32_entry_targets: List[Tuple[str, int]] = []
    init_array_labels: List[str] = []
    fini_array_labels: List[str] = []
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
        x32_entry_targets.append((lbl, _cdecl_va(sva)))
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

    # orig_to_new pairs for runtime __nm_icall lookup
    orig_to_new_pairs: List[Tuple[int, int]] = []
    for fva in sorted(analysis.functions):
        if fva in rewritten.orig_to_new:
            orig_to_new_pairs.append((fva, rewritten.orig_to_new[fva]))

    # R_386_COPY stdio slots (e.g. stderr in ET_EXEC)
    copy_stdio_slots: List[Tuple[str, int]] = []
    for r in analysis.rel_dyn:
        if r.r_type == 5 and r.sym_name in ("stdin", "stdout", "stderr"):
            copy_stdio_slots.append((r.sym_name, r.offset))

    # Build helper assembly with a gap for rewritten.code_bytes right after dispatch vector
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
        orig_elf_va=meta_va + orig_blob_off_in_meta,
        orig_elf_size=len(orig_blob_bytes),
        ref_va_for_bias=rx_va,
        init_array_labels=init_array_labels,
        fini_array_labels=fini_array_labels,
    )

    # Inject the rewritten code placeholder block immediately after __nm_dispatch_vec
    marker = "    .p2align 3\n\n" if False else "__nm_ensure_gs_init:"
    gap_asm = f"    .p2align 4\n__nm_rewritten_gap:\n    .skip {rewritten_len_aligned}\n\n"
    if marker not in raw_asm:
        raise PatcherError("internal error: __nm_ensure_gs_init marker missing")
    full_asm = raw_asm.replace(marker, gap_asm + marker, 1)

    # Preliminary defsyms to measure exact rx_sz, rodata_sz, rw_va
    def _make_defsyms(temp_rw_va: int, rx_end_est: int) -> Dict[str, int]:
        ds: Dict[str, int] = {
            "sym_text_new_start": rx_va,
            "sym_text_new_end": rx_end_est,
            "sym_orig_elf_blob": meta_va + orig_blob_off_in_meta,
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
        for idx, (_sname, got_slot_va, _slot_idx) in enumerate(rewritten.plt_entries):
            ds[f"sym_got_plt_{idx}"] = got_slot_va
        for sname, dst_va in copy_stdio_slots:
            ds[f"sym_copy_{sname}"] = dst_va

        # Extra GOT slots live inside ext_rw right after .dynamic (estimated reserve 0x400)
        got_base = temp_rw_va + 0x400
        for g_idx, gname in enumerate(extra_imports):
            ds[f"got_{gname}"] = got_base + g_idx * 4
        return ds

    # Pass A: assemble at placeholder rodata/data/bss VAs to measure exact .text & .rodata sizes
    t_bytes_a, r_bytes_a, _d_a, bss_sz_a, _ = assemble_and_link_helpers(
        full_asm,
        text_va=rx_va,
        rodata_va=rx_va + 0x400000,
        data_va=rx_va + 0x800000,
        bss_va=rx_va + 0xC00000,
        defsyms=_make_defsyms(rx_va + 0x800000, rx_va + 0x100000),
    )

    rx_sz = _align_up(len(t_bytes_a), 16)
    rodata_va = _align_up(rx_va + rx_sz, 0x1000)
    rodata_off = _align_up(rx_off + rx_sz, 0x1000)

    helpers_ro_sz = _align_up(len(r_bytes_a), 16)
    jt_base_va = rodata_va + helpers_ro_sz
    rodata_total_sz = helpers_ro_sz + _align_up(rewritten.rodata_jump_tables_size, 16)

    rw_va = _align_up(rodata_va + rodata_total_sz, 0x1000)
    rw_off = _align_up(rodata_off + rodata_total_sz, 0x1000)

    # Pass B: assemble & link at EXACT final VAs!
    # Reserve space at rw_va for .dynamic: up to 48 entries * 8 = 384 bytes -> reserve 0x400 bytes
    dynamic_reserve = 0x400
    extra_got_base_va = rw_va + dynamic_reserve
    extra_got_sz = len(extra_got_syms) * 4
    bss_va = _align_up(extra_got_base_va + extra_got_sz, 16)

    defsyms_final = _make_defsyms(rw_va, rx_va + rx_sz)
    for g_idx, gname in enumerate(extra_got_syms):
        defsyms_final[f"got_{gname}"] = extra_got_base_va + g_idx * 4
    if not need_dbt:
        defsyms_final["got_newminhh_dbt_enter_at"] = 0
        defsyms_final["got_newminhh_dbt_exec_embedded"] = 0

    t_bytes, r_bytes, _d_bytes, bss_sz, sym_vas = assemble_and_link_helpers(
        full_asm,
        text_va=rx_va,
        rodata_va=rodata_va,
        data_va=extra_got_base_va,
        bss_va=bss_va,
        defsyms=defsyms_final,
    )

    if len(t_bytes) > rx_sz:
        raise PatcherError(f"helper .text grew between pass A ({rx_sz}) and pass B ({len(t_bytes)})")

    # Splice rewritten.code_bytes into __nm_rewritten_gap and patch fixups
    rx_buf = bytearray(t_bytes)
    gap_start = dispatch_vec_sz
    rx_buf[gap_start : gap_start + len(rewritten.code_bytes)] = rewritten.code_bytes

    for patch_off, next_insn_va, jt_off in rewritten.rodata_fixups:
        target_va = jt_base_va + jt_off
        struct.pack_into("<i", rx_buf, gap_start + patch_off, target_va - next_insn_va)

    for patch_off, next_insn_va, lbl in rewritten.label_fixups:
        if lbl not in sym_vas:
            raise PatcherError(f"unresolved helper label '{lbl}' during ELF link")
        target_va = sym_vas[lbl]
        struct.pack_into("<i", rx_buf, gap_start + patch_off, target_va - next_insn_va)

    # 6. Build .rodata.newminhh (helpers rodata + relocated jump tables)
    rodata_buf = bytearray(r_bytes.ljust(helpers_ro_sz, b"\x00"))
    for jt_blob in rewritten.jump_table_blobs:
        jt_va = jt_base_va + jt_blob.rodata_offset
        for orig_t in jt_blob.targets:
            new_t = rewritten.orig_to_new[orig_t]
            rodata_buf.extend(struct.pack("<i", new_t - jt_va))
    while len(rodata_buf) < rodata_total_sz:
        rodata_buf.append(0)

    # 7. Patch original image bytes (e_machine, e_entry, ET_EXEC init/fini_array)
    orig_patched = bytearray(analysis.raw_bytes)
    struct.pack_into("<H", orig_patched, 18, EM_X86_64)

    if analysis.has_interp:
        new_entry = sym_vas["_nm_start"]
    else:
        new_entry = 0
    struct.pack_into("<I", orig_patched, 24, new_entry)

    init_arr_va = analysis.dyn_tags.get(DT_INIT_ARRAY, 0)
    init_arr_sz = analysis.dyn_tags.get(DT_INIT_ARRAYSZ, 0)
    fini_arr_va = analysis.dyn_tags.get(DT_FINI_ARRAY, 0)
    fini_arr_sz = analysis.dyn_tags.get(DT_FINI_ARRAYSZ, 0)

    init_arr_slot_map: Dict[int, int] = {}
    for i, lbl in enumerate(init_array_labels):
        init_arr_slot_map[init_arr_va + i * 4] = sym_vas[lbl]

    fini_arr_slot_map: Dict[int, int] = {}
    for i, lbl in enumerate(fini_array_labels):
        fini_arr_slot_map[fini_arr_va + i * 4] = sym_vas[lbl]

    if analysis.e_type == 2:  # ET_EXEC: overwrite .init_array / .fini_array slots directly
        for slot_va, shim_va in list(init_arr_slot_map.items()) + list(fini_arr_slot_map.items()):
            foff = _vaddr_to_offset(analysis.phdrs, slot_va)
            if foff is not None:
                struct.pack_into("<I", orig_patched, foff, shim_va)

    # Update .dynsym exports to point to x32->cdecl shims
    for sname, lbl in export_labels.items():
        didx = sym_name_to_idx[sname]
        new_dynsyms[didx].st_value = sym_vas[lbl]

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

    # 8. Build .rela.dyn and .rela.plt (DT_RELA)
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
        elif r.r_type == 6:  # R_386_GLOB_DAT -> R_X86_64_GLOB_DAT
            rela_dyn_bytes.extend(
                struct.pack("<IIi", r.offset, (r.sym_idx << 8) | R_X86_64_GLOB_DAT, 0)
            )
        elif r.r_type == 5:  # R_386_COPY -> R_X86_64_COPY
            rela_dyn_bytes.extend(
                struct.pack("<IIi", r.offset, (r.sym_idx << 8) | R_X86_64_COPY, 0)
            )
        elif r.r_type == 1:  # R_386_32 -> R_X86_64_32
            orig_val = _read_u32_at_va(analysis.phdrs, analysis.raw_bytes, r.offset)
            rela_dyn_bytes.extend(
                struct.pack("<IIi", r.offset, (r.sym_idx << 8) | R_X86_64_32, orig_val)
            )
        elif r.r_type == 2:  # R_386_PC32 -> R_X86_64_PC32
            orig_val = _read_u32_at_va(analysis.phdrs, analysis.raw_bytes, r.offset)
            rela_dyn_bytes.extend(
                struct.pack("<IIi", r.offset, (r.sym_idx << 8) | R_X86_64_PC32, orig_val)
            )
        else:
            raise PatcherError(f"unsupported i386 relocation type {r.r_type} at 0x{r.offset:08x}")

    for g_idx, gname in enumerate(extra_got_syms):
        slot_va = extra_got_base_va + g_idx * 4
        sidx = sym_name_to_idx[gname]
        rela_dyn_bytes.extend(struct.pack("<IIi", slot_va, (sidx << 8) | R_X86_64_GLOB_DAT, 0))

    rela_plt_bytes = bytearray()
    for r in analysis.rel_plt:
        rela_plt_bytes.extend(
            struct.pack("<IIi", r.offset, (r.sym_idx << 8) | R_X86_64_JUMP_SLOT, 0)
        )

    # 9. Assemble ext_ro_meta segment bytes
    hash_va = meta_va + hash_off_in_meta
    dynsym_va = meta_va + dynsym_off_in_meta
    dynstr_va = meta_va + dynstr_off_in_meta
    rela_dyn_va = meta_va + rela_dyn_off_in_meta
    rela_plt_va = meta_va + rela_plt_off_in_meta
    note_va = meta_va + note_off_in_meta
    interp_va = meta_va + interp_off_in_meta

    # Build .dynamic in ext_rw (at rw_va)
    dyn_entries: List[Tuple[int, int]] = []
    for noff in needed_str_offs:
        dyn_entries.append((DT_NEEDED, noff))
    if soname_dyn_off:
        dyn_entries.append((DT_SONAME, soname_dyn_off))
    dyn_entries.append((DT_INIT, sym_vas["_nm_init"]))
    dyn_entries.append((DT_FINI, sym_vas["_nm_fini"]))
    if init_arr_va and init_arr_sz:
        dyn_entries.append((DT_INIT_ARRAY, init_arr_va))
        dyn_entries.append((DT_INIT_ARRAYSZ, init_arr_sz))
    if fini_arr_va and fini_arr_sz:
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
        raise PatcherError("internal error: .dynamic exceeded dynamic_reserve")

    ext_rw_filesz = dynamic_reserve + extra_got_sz
    ext_rw_memsz = (bss_va - rw_va) + bss_sz
    ext_rw_buf = bytearray(ext_rw_filesz)
    ext_rw_buf[0 : len(dynamic_bytes)] = dynamic_bytes

    # 10. Build new Program Header Table (placed at meta_off / meta_va)
    final_phdrs: List[PhdrInfo] = []
    # Separate original non-LOAD phdrs and LOAD phdrs, update PT_PHDR / PT_INTERP / PT_DYNAMIC
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

    # Insert PT_LOADs right after PT_PHDR/PT_INTERP (standard ELF order)
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

    # Assemble ext_ro_meta buffer
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
    meta_buf[note_off_in_meta : note_off_in_meta + len(note_bytes)] = note_bytes
    meta_buf[orig_blob_off_in_meta : orig_blob_off_in_meta + len(orig_blob_bytes)] = orig_blob_bytes

    # 11. Build final image file & updated section headers
    out_buf = bytearray(orig_patched)
    out_buf.extend(b"\x00" * (meta_off - len(out_buf)))
    out_buf.extend(meta_buf)
    out_buf.extend(b"\x00" * (rx_off - len(out_buf)))
    out_buf.extend(rx_buf)
    out_buf.extend(b"\x00" * (rodata_off - len(out_buf)))
    out_buf.extend(rodata_buf)
    out_buf.extend(b"\x00" * (rw_off - len(out_buf)))
    out_buf.extend(ext_rw_buf)

    # Rebuild section header table (update dynamic/reloc/interp sections & append .newminhh sections)
    shstr_sec = None
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

    # Update original sections: turn old .rel.dyn / .rel.plt / .gnu.version* / .gnu.hash / .dynsym / .dynstr / .dynamic into SHT_PROGBITS .orig.*
    new_shdrs: List[Tuple[int, int, int, int, int, int, int, int, int, int]] = []
    if analysis.shdrs:
        orig_shstr_base = analysis.shdrs[e_shstrndx].sh_offset if e_shstrndx < len(analysis.shdrs) else 0
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

    # Append active x32 sections
    base_idx = len(new_shdrs)
    # Indices:
    # base_idx + 0: .interp (optional)
    # +1: .hash, +2: .dynsym, +3: .dynstr, +4: .rela.dyn, +5: .rela.plt, +6: .rodata.newminhh,
    # +7: .text.newminhh, +8: .dynamic, +9: .bss.newminhh, +10: .note.newminhh, +11: .newminhh.orig
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
    new_shdrs.append((_add_shstr(".rodata.newminhh"), 1, 2, rodata_va, rodata_off, len(rodata_buf), 0, 0, 16, 0))
    new_shdrs.append((_add_shstr(".text.newminhh"), 1, 6, rx_va, rx_off, len(rx_buf), 0, 0, 16, 0))
    new_shdrs.append((_add_shstr(".dynamic"), 6, 3, rw_va, rw_off, len(dynamic_bytes), dynstr_idx, 0, 4, 8))
    new_shdrs.append((_add_shstr(".data.newminhh"), 1, 3, extra_got_base_va, rw_off + dynamic_reserve, extra_got_sz, 0, 0, 4, 0))
    new_shdrs.append((_add_shstr(".bss.newminhh"), 8, 3, bss_va, rw_off + ext_rw_filesz, bss_sz, 0, 0, 16, 0))
    new_shdrs.append((_add_shstr(".note.newminhh"), 7, 2, note_va, meta_off + note_off_in_meta, len(note_bytes), 0, 0, 4, 0))
    new_shdrs.append((_add_shstr(".newminhh.orig"), 1, 2, meta_va + orig_blob_off_in_meta, meta_off + orig_blob_off_in_meta, len(orig_blob_bytes), 0, 0, 16, 0))

    new_shstr_idx = len(new_shdrs)
    new_shstr_off = len(out_buf)
    out_buf.extend(shstr_sec)
    new_shdrs.append((_add_shstr(".shstrtab.newminhh"), 3, 0, 0, new_shstr_off, len(shstr_sec), 0, 0, 1, 0))
    # Re-write shstr_sec if _add_shstr appended ".shstrtab.newminhh"
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
