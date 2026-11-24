"""Two-tier differential verifier for boosted -mx32 .o functions."""

from __future__ import annotations

import functools
import itertools
import os
import random
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

__all__ = [
    "DiffVerifyResult",
    "can_run_i386_native",
    "can_run_x32_native",
    "generate_test_vectors",
    "verify_boosted_function",
]

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
UINT32_MAX = (1 << 32) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
UINT64_MAX = (1 << 64) - 1

_NAN32 = struct.unpack("<f", struct.pack("<I", 0x7FC00000))[0]
_NAN64 = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000))[0]
_INF32 = struct.unpack("<f", struct.pack("<I", 0x7F800000))[0]
_INF64 = struct.unpack("<d", struct.pack("<Q", 0x7FF0000000000000))[0]
_MIN_SUBNORM32 = struct.unpack("<f", struct.pack("<I", 1))[0]
_MAX_SUBNORM32 = struct.unpack("<f", struct.pack("<I", 0x7FFFFF))[0]
_MIN_SUBNORM64 = struct.unpack("<d", struct.pack("<Q", 1))[0]
_MAX_SUBNORM64 = struct.unpack("<d", struct.pack("<Q", 0xFFFFFFFFFFFFF))[0]

_INT_RANGES = {
    "i32": (INT32_MIN, INT32_MAX),
    "u32": (0, UINT32_MAX),
    "i64": (INT64_MIN, INT64_MAX),
    "u64": (0, UINT64_MAX),
}

_BOUNDARY_VALUES = {
    "i32": [0, 1, -1, INT32_MIN, INT32_MAX],
    "u32": [0, 1, UINT32_MAX, INT32_MAX, 0x80000000],
    "i64": [0, 1, -1, INT64_MIN, INT64_MAX],
    "u64": [0, 1, UINT64_MAX, INT64_MAX, 0x8000000000000000],
    "f32": [0.0, -0.0, 1.0, -1.0, _NAN32, _INF32, -_INF32, _MIN_SUBNORM32, _MAX_SUBNORM32],
    "f64": [0.0, -0.0, 1.0, -1.0, _NAN64, _INF64, -_INF64, _MIN_SUBNORM64, _MAX_SUBNORM64],
    "void": [0],
}

_C_TYPE_MAP = {
    "i32": "nm_i32",
    "u32": "nm_u32",
    "i64": "nm_i64",
    "u64": "nm_u64",
    "f32": "nm_f32",
    "f64": "nm_f64",
    "void": "void",
}


@dataclass
class DiffVerifyResult:
    """Outcome of the differential verification of a boosted function."""

    passed: bool
    mode: str
    trials_run: int
    first_mismatch: Optional[str]
    reason: str


def _as_bits(value: object, type_str: str) -> int:
    if type_str == "f32":
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]  # type: ignore[arg-type]
    if type_str == "f64":
        return struct.unpack("<Q", struct.pack("<d", float(value)))[0]  # type: ignore[arg-type]
    return int(value)  # type: ignore[arg-type]


def _rand_value(rng: random.Random, type_str: str) -> int:
    if type_str in _INT_RANGES:
        lo, hi = _INT_RANGES[type_str]
        return rng.randint(lo, hi)
    if type_str == "f32":
        return rng.getrandbits(32)
    if type_str == "f64":
        return rng.getrandbits(64)
    return 0


def generate_test_vectors(
    arg_types: List[str], trials: int = 4096, seed: int = 0x4E4D3341
) -> List[Tuple[int, ...]]:
    """Generate deterministic test vectors: boundary cases first, then PRNG."""
    types = [str(t) for t in arg_types]
    rng = random.Random(seed)
    n_args = len(types)

    if trials <= 0:
        return []
    if n_args == 0:
        return [() for _ in range(trials)]

    boundary_lists = [
        [_as_bits(v, t) for v in _BOUNDARY_VALUES.get(t, [0])]
        for t in types
    ]

    vectors: List[Tuple[int, ...]] = []
    seen = set()

    # Diagonal combinations so every argument slot hits every boundary value early.
    max_b = max(len(bl) for bl in boundary_lists)
    for idx in range(max_b):
        vec = tuple(bl[idx % len(bl)] for bl in boundary_lists)
        if vec not in seen:
            seen.add(vec)
            vectors.append(vec)

    # Cartesian prefix up to 64 boundary combinations.
    for combo in itertools.islice(itertools.product(*boundary_lists), 64):
        if combo not in seen:
            seen.add(combo)
            vectors.append(combo)

    while len(vectors) < trials:
        vectors.append(tuple(_rand_value(rng, t) for t in types))

    return vectors[:trials]


_I386_PROBE_C = r"""
__asm__(
".globl _start\n"
"_start:\n"
"    and $-16, %esp\n"
"    call probe_main\n"
"    mov %eax, %ebx\n"
"    mov $252, %eax\n"
"    int $0x80\n"
"    hlt\n"
".globl i386_mmap2\n"
"i386_mmap2:\n"
"    push %ebx\n"
"    push %esi\n"
"    push %edi\n"
"    push %ebp\n"
"    mov 20(%esp), %ebx\n"
"    mov 24(%esp), %ecx\n"
"    mov 28(%esp), %edx\n"
"    mov 32(%esp), %esi\n"
"    mov 36(%esp), %edi\n"
"    mov 40(%esp), %ebp\n"
"    mov $192, %eax\n"
"    int $0x80\n"
"    pop %ebp\n"
"    pop %edi\n"
"    pop %esi\n"
"    pop %ebx\n"
"    ret\n"
".globl i386_mprotect\n"
"i386_mprotect:\n"
"    push %ebx\n"
"    mov 8(%esp), %ebx\n"
"    mov 12(%esp), %ecx\n"
"    mov 16(%esp), %edx\n"
"    mov $125, %eax\n"
"    int $0x80\n"
"    pop %ebx\n"
"    ret\n"
);

long i386_mmap2(unsigned long, unsigned long, unsigned long, unsigned long, long, unsigned long);
int i386_mprotect(unsigned long, unsigned long, int);

int probe_main(void) {
    unsigned long addr = (unsigned long)i386_mmap2(0x10000000UL, 4096, 3, 0x100022UL, -1, 0);
    if (addr >= (unsigned long)-4095UL) {
        return 1;
    }
    *((unsigned char *)addr) = 0xC3; /* ret */
    if (i386_mprotect(addr, 4096, 5) != 0) {
        return 2;
    }
    ((void (*)(void))addr)();
    return 0;
}
"""

_X32_PROBE_C = r"""
__asm__(
".globl _start\n"
"_start:\n"
"    call probe_main\n"
"    mov %eax, %edi\n"
"    mov $0x400000E7, %rax\n"
"    syscall\n"
"    hlt\n"
);

int probe_main(void) {
    unsigned long long ret = 0;
    const char buf[1] = {0};
    unsigned long long ptr = (unsigned long long)(unsigned long)buf;
    __asm__ volatile(
        "mov $0x40000001, %%rax\n"
        "syscall\n"
        : "=a"(ret)
        : "D"(1), "S"(ptr), "d"(0ULL)
        : "rcx", "r11", "memory"
    );
    return (ret > (unsigned long long)-4095LL) ? 1 : 0;
}
"""


def _run(cmd: List[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)


def _freestanding_flags(arch_flag: str) -> List[str]:
    return [
        "gcc",
        arch_flag,
        "-ffreestanding",
        "-fno-stack-protector",
        "-fcf-protection=none",
        "-nostdlib",
        "-static",
        "-no-pie",
        "-O2",
    ]


@functools.lru_cache(maxsize=None)
def can_run_i386_native() -> bool:
    """True when a freestanding gcc -m32 probe compiles and executes cleanly."""
    with tempfile.TemporaryDirectory(prefix="diffverify_probe_i386_") as td:
        src = os.path.join(td, "probe.c")
        exe = os.path.join(td, "probe")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(_I386_PROBE_C)
        proc = _run(_freestanding_flags("-m32") + [src, "-o", exe], timeout=30)
        if proc.returncode != 0 or not os.path.exists(exe):
            return False
        try:
            return _run([exe], timeout=10).returncode == 0
        except Exception:
            return False


@functools.lru_cache(maxsize=None)
def can_run_x32_native() -> bool:
    """True when a freestanding gcc -mx32 probe compiles and executes cleanly."""
    with tempfile.TemporaryDirectory(prefix="diffverify_probe_x32_") as td:
        src = os.path.join(td, "probe.c")
        exe = os.path.join(td, "probe")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(_X32_PROBE_C)
        proc = _run(_freestanding_flags("-mx32") + [src, "-o", exe], timeout=30)
        if proc.returncode != 0 or not os.path.exists(exe):
            return False
        try:
            return _run([exe], timeout=10).returncode == 0
        except Exception:
            return False


def _write_source(directory: str, name: str, source: str) -> str:
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(source)
    return path


def _c_type(t: str) -> str:
    return _C_TYPE_MAP.get(t, "nm_i32")


def _common_typedefs() -> str:
    return (
        "typedef int nm_i32;\n"
        "typedef unsigned int nm_u32;\n"
        "typedef long long nm_i64;\n"
        "typedef unsigned long long nm_u64;\n"
        "typedef float nm_f32;\n"
        "typedef double nm_f64;\n"
        "union nm_bits { nm_u64 u64; nm_u32 u32; nm_i32 i32; nm_i64 i64; nm_f32 f32; nm_f64 f64; };\n"
    )


def _gen_vectors_c(vectors: List[Tuple[int, ...]], arg_types: List[str]) -> str:
    n_vecs = len(vectors)
    n_cols = len(arg_types)
    if n_cols == 0:
        return f"#define VEC_COUNT {n_vecs}\n"

    lines = [f"const unsigned long long g_vecs[{n_vecs}][{n_cols}] = {{"]
    for vec in vectors:
        cells = []
        for val, t in zip(vec, arg_types):
            if t in ("i64", "u64", "f64"):
                cells.append(f"0x{val & UINT64_MAX:016x}ULL")
            else:
                cells.append(f"0x{val & UINT32_MAX:08x}ULL")
        lines.append("    {" + ", ".join(cells) + "},")
    lines.append("};")
    lines.append(f"#define VEC_COUNT {n_vecs}")
    return "\n".join(lines) + "\n"


def _gen_call_expr(fn_expr: str, ret_type: str, arg_types: List[str]) -> str:
    nargs = len(arg_types)
    param_types = ", ".join(_c_type(t) for t in arg_types) if nargs else "void"
    ret_c = _c_type(ret_type)

    lines = [f"typedef {ret_c} (*nm_fn_t)({param_types});", f"nm_fn_t fn = (nm_fn_t)({fn_expr});"]
    call_args: List[str] = []
    for idx, t in enumerate(arg_types):
        lines.append(f"union nm_bits b{idx}; b{idx}.u64 = g_vecs[i][{idx}];")
        field = {
            "i32": "i32",
            "u32": "u32",
            "i64": "i64",
            "u64": "u64",
            "f32": "f32",
            "f64": "f64",
        }.get(t, "u32")
        call_args.append(f"b{idx}.{field}")

    args_str = ", ".join(call_args)
    if ret_type == "void":
        lines.append(f"fn({args_str});")
        lines.append("nm_u64 rec = 0ULL;")
    else:
        lines.append(f"{ret_c} r = fn({args_str});")
        lines.append("union nm_bits rb; rb.u64 = 0ULL;")
        if ret_type in ("i32", "u32"):
            lines.append("rb.u32 = (nm_u32)r;")
        elif ret_type == "f32":
            lines.append("rb.f32 = r;")
        elif ret_type == "f64":
            lines.append("rb.f64 = r;")
        else:
            lines.append("rb.u64 = (nm_u64)r;")
        lines.append("nm_u64 rec = rb.u64;")
    return "\n        ".join(lines)


def _gen_i386_harness(
    orig_va: int,
    orig_bytes: bytes,
    n_vecs: int,
    ret_type: str,
    arg_types: List[str],
) -> str:
    nargs = len(arg_types)
    bytes_lit = ", ".join(f"0x{b:02x}" for b in orig_bytes)
    page_off = orig_va & 0xFFF
    map_len = max(4096, ((page_off + len(orig_bytes) + 0xFFF) & ~0xFFF))
    call_block = _gen_call_expr("g_page", ret_type, arg_types)

    vecs_extern = f"extern const nm_u64 g_vecs[][{nargs}];\n" if nargs else ""
    return f"""/* Auto-generated i386 differential harness (diffverify.py). */
#define VEC_COUNT {n_vecs}U
{_common_typedefs()}
{vecs_extern}
static const unsigned long g_orig_va = 0x{orig_va:x}UL;
static const unsigned long g_page_off = 0x{page_off:x}UL;
static const unsigned long g_map_len = 0x{map_len:x}UL;
static const unsigned char g_orig_bytes[] = {{{bytes_lit}}};
static void *g_page;
static nm_u64 g_outbuf[512];

__asm__(
".globl _start\\n"
"_start:\\n"
"    and $-16, %esp\\n"
"    call run_harness\\n"
"    push %eax\\n"
"    call i386_exit_group\\n"
".globl i386_mmap2\\n"
"i386_mmap2:\\n"
"    push %ebx\\n"
"    push %esi\\n"
"    push %edi\\n"
"    push %ebp\\n"
"    mov 20(%esp), %ebx\\n"
"    mov 24(%esp), %ecx\\n"
"    mov 28(%esp), %edx\\n"
"    mov 32(%esp), %esi\\n"
"    mov 36(%esp), %edi\\n"
"    mov 40(%esp), %ebp\\n"
"    mov $192, %eax\\n"
"    int $0x80\\n"
"    pop %ebp\\n"
"    pop %edi\\n"
"    pop %esi\\n"
"    pop %ebx\\n"
"    ret\\n"
".globl i386_mprotect\\n"
"i386_mprotect:\\n"
"    push %ebx\\n"
"    mov 8(%esp), %ebx\\n"
"    mov 12(%esp), %ecx\\n"
"    mov 16(%esp), %edx\\n"
"    mov $125, %eax\\n"
"    int $0x80\\n"
"    pop %ebx\\n"
"    ret\\n"
".globl i386_write\\n"
"i386_write:\\n"
"    push %ebx\\n"
"    mov 8(%esp), %ebx\\n"
"    mov 12(%esp), %ecx\\n"
"    mov 16(%esp), %edx\\n"
"    mov $4, %eax\\n"
"    int $0x80\\n"
"    pop %ebx\\n"
"    ret\\n"
".globl i386_exit_group\\n"
"i386_exit_group:\\n"
"    mov 4(%esp), %ebx\\n"
"    mov $252, %eax\\n"
"    int $0x80\\n"
"    hlt\\n"
);

unsigned long i386_mmap2(unsigned long, unsigned long, unsigned long, unsigned long, long, unsigned long);
int i386_mprotect(unsigned long, unsigned long, int);
long i386_write(int, const void *, unsigned long);
void i386_exit_group(int);

static void flush_buf(unsigned int count) {{
    const unsigned char *p = (const unsigned char *)g_outbuf;
    unsigned long rem = (unsigned long)count * 8UL;
    while (rem > 0) {{
        long w = i386_write(1, p, rem);
        if (w <= 0) i386_exit_group(4);
        p += w;
        rem -= (unsigned long)w;
    }}
}}

static int i386_setup(void) {{
    unsigned long flags = 0x100022UL; /* MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE */
    unsigned long req = (g_orig_va < 0x10000UL) ? 0x10000000UL : (g_orig_va & ~0xFFFUL);
    unsigned long addr = i386_mmap2(req, g_map_len, 3, flags, -1, 0);
    if (addr >= (unsigned long)-4095UL && req != 0x10000000UL) {{
        addr = i386_mmap2(0x10000000UL, g_map_len, 3, flags, -1, 0);
    }}
    if (addr >= (unsigned long)-4095UL) return -1;
    unsigned char *dst = (unsigned char *)(addr + g_page_off);
    for (unsigned long k = 0; k < sizeof(g_orig_bytes); k++) {{
        dst[k] = g_orig_bytes[k];
    }}
    if (i386_mprotect(addr, g_map_len, 5) != 0) return -1;
    g_page = (void *)dst;
    return 0;
}}

int run_harness(void) {{
    if (i386_setup() != 0) return 3;
    unsigned int buf_idx = 0;
    for (unsigned int i = 0; i < VEC_COUNT; i++) {{
        {call_block}
        g_outbuf[buf_idx++] = rec;
        if (buf_idx == 512) {{
            flush_buf(buf_idx);
            buf_idx = 0;
        }}
    }}
    if (buf_idx > 0) flush_buf(buf_idx);
    return 0;
}}
"""


def _gen_x32_harness(
    symbol_name: str,
    n_vecs: int,
    ret_type: str,
    arg_types: List[str],
) -> str:
    nargs = len(arg_types)
    param_types = ", ".join(_c_type(t) for t in arg_types) if nargs else "void"
    ret_c = _c_type(ret_type)
    call_block = _gen_call_expr(symbol_name, ret_type, arg_types)
    vecs_extern = f"extern const nm_u64 g_vecs[][{nargs}];\n" if nargs else ""

    return f"""/* Auto-generated x32 differential harness (diffverify.py). */
#define VEC_COUNT {n_vecs}U
{_common_typedefs()}
{vecs_extern}
extern {ret_c} {symbol_name}({param_types});
static nm_u64 g_outbuf[512];

__asm__(
".globl _start\\n"
"_start:\\n"
"    call run_harness\\n"
"    mov %eax, %edi\\n"
"    mov $0x400000E7, %rax\\n"
"    syscall\\n"
"    hlt\\n"
);

static long x32_write(int fd, const void *buf, unsigned long count) {{
    long long ret;
    unsigned long long ptr = (unsigned long long)(unsigned long)buf;
    __asm__ volatile(
        "mov $0x40000001, %%rax\\n"
        "syscall\\n"
        : "=a"(ret)
        : "D"(fd), "S"(ptr), "d"((unsigned long long)count)
        : "rcx", "r11", "memory"
    );
    return (long)ret;
}}

static void x32_exit_group(int code) {{
    __asm__ volatile(
        "mov $0x400000E7, %%rax\\n"
        "syscall\\n"
        :
        : "D"(code)
        : "rax", "rcx", "r11"
    );
    __builtin_unreachable();
}}

static void flush_buf(unsigned int count) {{
    const unsigned char *p = (const unsigned char *)g_outbuf;
    unsigned long rem = (unsigned long)count * 8UL;
    while (rem > 0) {{
        long w = x32_write(1, p, rem);
        if (w <= 0) x32_exit_group(4);
        p += w;
        rem -= (unsigned long)w;
    }}
}}

int run_harness(void) {{
    unsigned int buf_idx = 0;
    for (unsigned int i = 0; i < VEC_COUNT; i++) {{
        {call_block}
        g_outbuf[buf_idx++] = rec;
        if (buf_idx == 512) {{
            flush_buf(buf_idx);
            buf_idx = 0;
        }}
    }}
    if (buf_idx > 0) flush_buf(buf_idx);
    return 0;
}}
"""


def _detect_l2a_symbol(obj_path: str, func_name: str) -> str:
    candidates = (f"nm_l3_{func_name}", f"nm_l2a_{func_name}", func_name)
    proc = _run(["nm", "-g", "--defined-only", obj_path], timeout=10)
    if proc.returncode == 0:
        syms = {
            line.split()[-1]
            for line in proc.stdout.decode(errors="replace").splitlines()
            if line.strip()
        }
        for cand in candidates:
            if cand in syms:
                return cand
    return f"nm_l3_{func_name}"


def _is_nan_record(buf: bytes, ret_type: str) -> bool:
    try:
        if ret_type == "f32":
            v = struct.unpack("<f", buf[:4])[0]
            return v != v
        if ret_type == "f64":
            v = struct.unpack("<d", buf[:8])[0]
            return v != v
    except Exception:
        pass
    return False


def _compare_buffers(
    expected: bytes, actual: bytes, vectors: List[Tuple[int, ...]], ret_type: str
) -> Optional[str]:
    rec = 8
    n = min(len(expected), len(actual), len(vectors) * rec)
    for off in range(0, n, rec):
        exp_rec = expected[off:off + rec]
        act_rec = actual[off:off + rec]
        if exp_rec == act_rec:
            continue
        if _is_nan_record(exp_rec, ret_type) and _is_nan_record(act_rec, ret_type):
            continue
        idx = off // rec
        inputs = vectors[idx] if idx < len(vectors) else ()
        inputs_str = ", ".join(hex(v) for v in inputs)
        exp_val = struct.unpack("<Q", exp_rec)[0]
        act_val = struct.unpack("<Q", act_rec)[0]
        return (
            f"trial {idx}: inputs=({inputs_str}): "
            f"expected 0x{exp_val:016x} got 0x{act_val:016x}"
        )
    if len(expected) != len(actual) or len(expected) != len(vectors) * rec:
        return (
            f"length mismatch: expected {len(vectors) * rec} bytes, "
            f"got ref={len(expected)} actual={len(actual)}"
        )
    return None


def verify_boosted_function(
    func_name: str,
    orig_va: int,
    orig_bytes: bytes,
    boosted_obj_path: str,
    ret_type: str,
    arg_types: List[str],
    *,
    l2a_ref_asm: Optional[str] = None,
    trials: int = 4096,
    timeout: int = 30,
) -> DiffVerifyResult:
    """Two-tier differential verification of a boosted -mx32 function.

    Mode 1 (``host-i386-vs-x32``): executes ``orig_bytes`` in a freestanding
    32-bit harness and ``nm_l3_<func_name>`` from ``boosted_obj_path`` in an
    x32 harness across deterministic boundary + PRNG test vectors.

    Mode 2 (``x32-l2a-vs-l3``): fallback when native i386 execution is unavailable
    and ``l2a_ref_asm`` is supplied.

    Fails closed when neither tier can execute.
    """
    i386_ok = can_run_i386_native()
    x32_ok = can_run_x32_native()

    if i386_ok and x32_ok:
        mode = "host-i386-vs-x32"
    elif x32_ok and not i386_ok and l2a_ref_asm:
        mode = "x32-l2a-vs-l3"
    else:
        return DiffVerifyResult(
            passed=False,
            mode="unavailable",
            trials_run=0,
            first_mismatch=None,
            reason=(
                f"no execution tier available: i386={i386_ok} x32={x32_ok} "
                f"l2a_ref_asm={'provided' if l2a_ref_asm else 'missing'}"
            ),
        )

    if not os.path.isfile(boosted_obj_path):
        return DiffVerifyResult(
            passed=False,
            mode=mode,
            trials_run=0,
            first_mismatch=None,
            reason=f"boosted object not found: {boosted_obj_path}",
        )

    vectors = generate_test_vectors(arg_types, trials=trials)
    n_trials = len(vectors)
    target_sym = f"nm_l3_{func_name}"

    with tempfile.TemporaryDirectory(prefix="diffverify_") as td:
        vecs_path = _write_source(td, "vecs.c", _gen_vectors_c(vectors, arg_types))
        x32_path = _write_source(td, "x32_harness.c", _gen_x32_harness(target_sym, n_trials, ret_type, arg_types))
        out_x32 = os.path.join(td, "x32_run")

        comp_x32 = _run(
            _freestanding_flags("-mx32") + [x32_path, vecs_path, boosted_obj_path, "-o", out_x32],
            timeout,
        )
        if comp_x32.returncode != 0:
            return DiffVerifyResult(
                passed=False,
                mode=mode,
                trials_run=0,
                first_mismatch=None,
                reason=f"x32 harness compile failed: {comp_x32.stderr.decode(errors='replace')[:500]}",
            )

        if mode == "host-i386-vs-x32":
            i386_path = _write_source(
                td,
                "i386_harness.c",
                _gen_i386_harness(orig_va, orig_bytes, n_trials, ret_type, arg_types),
            )
            out_i32 = os.path.join(td, "i386_run")
            comp_i32 = _run(
                _freestanding_flags("-m32") + [i386_path, vecs_path, "-o", out_i32],
                timeout,
            )
            if comp_i32.returncode != 0:
                return DiffVerifyResult(
                    passed=False,
                    mode=mode,
                    trials_run=0,
                    first_mismatch=None,
                    reason=f"i386 harness compile failed: {comp_i32.stderr.decode(errors='replace')[:500]}",
                )
            ref_run = _run([out_i32], timeout)
            if ref_run.returncode != 0:
                return DiffVerifyResult(
                    passed=False,
                    mode=mode,
                    trials_run=0,
                    first_mismatch=None,
                    reason=f"i386 harness run failed rc={ref_run.returncode}",
                )
        else:
            assert l2a_ref_asm is not None
            asm_path = _write_source(td, "l2a_ref.s", l2a_ref_asm)
            l2a_obj = os.path.join(td, "l2a.o")
            asm_proc = _run(["as", "--x32", asm_path, "-o", l2a_obj], timeout)
            if asm_proc.returncode != 0:
                return DiffVerifyResult(
                    passed=False,
                    mode=mode,
                    trials_run=0,
                    first_mismatch=None,
                    reason=f"assembling l2a_ref_asm failed: {asm_proc.stderr.decode(errors='replace')[:500]}",
                )
            l2a_sym = _detect_l2a_symbol(l2a_obj, func_name)
            l2a_path = _write_source(td, "l2a_harness.c", _gen_x32_harness(l2a_sym, n_trials, ret_type, arg_types))
            out_l2a = os.path.join(td, "l2a_run")
            comp_l2a = _run(
                _freestanding_flags("-mx32") + [l2a_path, vecs_path, l2a_obj, "-o", out_l2a],
                timeout,
            )
            if comp_l2a.returncode != 0:
                return DiffVerifyResult(
                    passed=False,
                    mode=mode,
                    trials_run=0,
                    first_mismatch=None,
                    reason=f"l2a harness link failed: {comp_l2a.stderr.decode(errors='replace')[:500]}",
                )
            ref_run = _run([out_l2a], timeout)
            if ref_run.returncode != 0:
                return DiffVerifyResult(
                    passed=False,
                    mode=mode,
                    trials_run=0,
                    first_mismatch=None,
                    reason=f"l2a harness run failed rc={ref_run.returncode}",
                )

        act_run = _run([out_x32], timeout)
        if act_run.returncode != 0:
            return DiffVerifyResult(
                passed=False,
                mode=mode,
                trials_run=0,
                first_mismatch=None,
                reason=f"x32 harness run failed rc={act_run.returncode}",
            )

        mismatch = _compare_buffers(ref_run.stdout, act_run.stdout, vectors, ret_type)
        if mismatch is not None:
            return DiffVerifyResult(
                passed=False,
                mode=mode,
                trials_run=n_trials,
                first_mismatch=mismatch,
                reason="differential mismatch found",
            )
        return DiffVerifyResult(
            passed=True,
            mode=mode,
            trials_run=n_trials,
            first_mismatch=None,
            reason="",
        )


def _run_self_tests() -> bool:
    print("[diffverify] Running inline self-tests...")
    vecs = generate_test_vectors(["i32", "i32"], trials=200)
    assert len(vecs) == 200
    assert vecs[0] == (0, 0)
    assert any(-1 in v for v in vecs[:20])
    assert any(INT32_MIN in v for v in vecs[:20])
    assert vecs == generate_test_vectors(["i32", "i32"], trials=200)

    assert _compare_buffers(b"\x01\x00\x00\x00\x00\x00\x00\x00", b"\x01\x00\x00\x00\x00\x00\x00\x00", [(1,)], "i32") is None
    mm = _compare_buffers(b"\x00" * 8, b"\x01" * 8, [(5,)], "i32")
    assert mm is not None and "trial 0" in mm

    i386_ok = can_run_i386_native()
    x32_ok = can_run_x32_native()
    print(f"  [INFO] can_run_i386_native()={i386_ok} can_run_x32_native()={x32_ok}")
    print("[diffverify] Inline self-tests passed!")
    return True


if __name__ == "__main__":
    _run_self_tests()
