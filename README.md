# newminhh/patcher

Containerized (`re-env:latest`) hybrid static binary patcher (`newminhh-patch` / `re_patcher`) for the [`newminhh`](https://github.com/newminhh/newminhh) i386-on-x32 compatibility system.

Transforms i386 `ELF32` (`EM_386`) executables and shared libraries into native x32 `ELF32` (`EM_X86_64`) binaries using `/libx32/ld-linux-x32.so.2` and `libnewminhh_compat.so` ([`newminhh/compat`](https://github.com/newminhh/compat)), with automatic or explicit per-function hybrid DBT fallback ([`newminhh/dbt`](https://github.com/newminhh/dbt)).

The patcher implements a Multi-Level Patcher pipeline: **Level 1** (lookup-driven static byte rewriting), **Level 2A** (CFG-driven anchor-preserving symbolic rewrite, default after verification), gated **Level 2B** (full relocatable PIE rebuild), and gated **Level 3A** (selective freestanding recompile boost) + **3C `--decompile-assist`** with `rz-ghidra`, Ghidra, IDA Pro, and Universal IDA Free backends. It consumes layered [`newminhh/lookup`](../lookup/README.md) ABI classification/registry data and follows multi-distro glibc+musl x32 trigger/cache precedence where lookup resolution applies.

---

## Files

| File | Role |
| :--- | :--- |
| `newminhh-patch` | Host CLI wrapper invoking Docker `re-env:latest` (`NEWMINHH_RE_IMAGE`) as the current UID/GID with input/output volume mounts |
| `re_patcher/__init__.py` | Package metadata and fail-loudly `PatcherError` exception |
| `re_patcher/cli.py` | **Stage 0**: Strict ELF32/i386 & `PT_DYNAMIC` structural validator (`validate_elf32_i386`) and CLI orchestration |
| `re_patcher/analyzer.py` | **Stage 1**: Multi-tool RE CFG/function/jump-table/PIC-thunk/syscall analyzer (`elftools`, `capstone`, `angr` `CFGFast`, `radare2`, optional `Ghidra`), N07/N08 forbidden-instruction checker, and `REWRITABLE_NATIVE` vs `REQUIRES_DBT_FALLBACK` classifier |
| `re_patcher/rewriter.py` | **Stage 2b**: Out-of-place i386 -> Long Mode x32 instruction rewriter preserving a 4-byte cdecl stack stride on native `%rsp` |
| `re_patcher/trampolines.py` | **Stage 2a**: Generates x32 GAS assembly (compiled via `as --x32` + `ld -m elf32_x86_64`) for `__nm_dispatch_vec`, `__nm_ensure_gs_init`, `_nm_init`/`_nm_fini`, `_nm_start`, `__nm_dual_abi_common`, `__nm_icall`/`__nm_ijmp`, `__nm_syscall_tramp`, `__nm_plt_*`, and `__nm_fb_*` hybrid DBT stubs |
| `re_patcher/elf_builder.py` | **Stage 2c**: Emits final `EM_X86_64` `ELF32` binary with `DT_RELA`, `DF_BIND_NOW`, `.note.newminhh`, `.newminhh.orig`, `.text.newminhh`, `.rodata.newminhh`, and 4 new `PT_LOAD` segments |

---

## CLI Usage

```bash
./newminhh-patch [--hybrid-fallback | --no-hybrid-fallback] [--use-ghidra] [--use-r2] [-v] <input-i386-elf> -o <output-x32-elf>
```

### Flags

| Flag | Behavior |
| :--- | :--- |
| `-o, --output <path>` | Output x32 ELF path (required; must be distinct from input; stale outputs are deleted on failure) |
| *(default: auto)* | Emits native rewritten code; links `libnewminhh_dbt.so` and `__nm_fb_*` stubs only if any function is classified `REQUIRES_DBT_FALLBACK` |
| `--hybrid-fallback` | Forces hybrid mode: always embeds `.newminhh.orig` and links `libnewminhh_dbt.so` (`newminhh_dbt_enter_at` / `newminhh_dbt_exec_embedded`) |
| `--no-hybrid-fallback` | Pure static mode: never links `libnewminhh_dbt.so`; fails loudly if any function requires DBT fallback |
| `--use-r2` | Corroborates CFG/function discovery with `radare2` (`aaa; aflj`); auto-enabled when `.symtab` is missing |
| `--use-ghidra` | Corroborates function discovery with `/opt/ghidra/support/analyzeHeadless` |
| `-v, --verbose` | Prints analysis and rewrite summary statistics |

### Environment Variables

- `NEWMINHH_RE_IMAGE`: Docker image containing Python RE dependencies (`elftools`, `capstone`, `angr`) and x32 binutils (`as --x32`, `ld -m elf32_x86_64`); defaults to `re-env:latest`.

---

## Lookup & Multi-Level Patcher

- **Lookup integration**: Uses [`src/lookup`](../lookup/README.md) layered `lookup_table.txt` precedence and generated Python registry (`re_patcher/_lookup_registry_gen.py`) for ABI classification.
- **Levels**: Level 1 / Level 2A / gated Level 2B / gated Level 3A + 3C `--decompile-assist`.
- **Decompile-assist backends**: `rz-ghidra`, Ghidra, IDA Pro, and Universal IDA Free (via export/import workflow).

## Documentation

- [`docs/ARCHITECTURE.md`](https://github.com/newminhh/docs/blob/main/ARCHITECTURE.md) — Detailed walkthrough of the 3-stage patching pipeline, 4-byte stack-stride encoding rules, dual-ABI dispatch, and ELF segment layout
- [`docs/patcher_levels.md`](https://github.com/newminhh/docs/blob/main/patcher_levels.md) — Multi-Level Patcher levels, gates, decompiler backends, and CLI flags
- [`docs/lookup_table.md`](https://github.com/newminhh/docs/blob/main/lookup_table.md) — Lookup schema, merge precedence, and ABI classification contracts
- [`docs/distro_triggers.md`](https://github.com/newminhh/docs/blob/main/distro_triggers.md) — Multi-distro glibc+musl x32 trigger and lookup regeneration policy
- [`docs/TESTING_AND_VM.md`](https://github.com/newminhh/docs/blob/main/TESTING_AND_VM.md) — Part 3 static/hybrid patcher test suite and N01–N09 negative fixture checks
