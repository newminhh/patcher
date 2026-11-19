"""Stage 2a — Long Mode x32 Trampolines, Compat Signatures, & Assembler/Linker Driver."""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from elftools.elf.elffile import ELFFile

from . import PatcherError

# Match src/common/newminhh_abi.h
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

# Symbol name -> (ret_kind, [arg_kinds], is_variadic, is_cdecl_direct)
COMPAT_SIGNATURES: Dict[str, Tuple[int, List[int], int, bool]] = {
    # Alloc / mem
    "malloc": (NM_RET_PTR, [NM_ARG_U32], 0, False),
    "calloc": (NM_RET_PTR, [NM_ARG_U32, NM_ARG_U32], 0, False),
    "realloc": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_U32], 0, False),
    "free": (NM_RET_VOID, [NM_ARG_PTR], 0, False),
    "memset": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_I32, NM_ARG_U32], 0, False),
    "memcpy": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_U32], 0, False),
    "memmove": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_U32], 0, False),
    "memcmp": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_U32], 0, False),
    # Strings & numeric
    "strlen": (NM_RET_U32, [NM_ARG_PTR], 0, False),
    "strcmp": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "strncmp": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_U32], 0, False),
    "strcpy": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "strncpy": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_U32], 0, False),
    "strcat": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "strchr": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_I32], 0, False),
    "strrchr": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_I32], 0, False),
    "strstr": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "strdup": (NM_RET_PTR, [NM_ARG_PTR], 0, False),
    "strtol": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_I32], 0, False),
    "strtoul": (NM_RET_U32, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_I32], 0, False),
    "strtoll": (NM_RET_I64, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_I32], 0, False),
    "strtoull": (NM_RET_U64, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_I32], 0, False),
    "__isoc23_strtol": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_I32], 0, False),
    "__isoc23_strtoul": (NM_RET_U32, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_I32], 0, False),
    "__isoc23_strtoll": (NM_RET_I64, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_I32], 0, False),
    "__isoc23_strtoull": (NM_RET_U64, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_I32], 0, False),
    "atoi": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "atol": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "getenv": (NM_RET_PTR, [NM_ARG_PTR], 0, False),
    "setenv": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_I32], 0, False),
    "unsetenv": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    # Stdio
    "printf": (NM_RET_I32, [NM_ARG_PTR], 1, True),
    "fprintf": (NM_RET_I32, [NM_ARG_PTR], 1, True),
    "sprintf": (NM_RET_I32, [NM_ARG_PTR], 1, True),
    "snprintf": (NM_RET_I32, [NM_ARG_PTR], 1, True),
    "vprintf": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "vfprintf": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "vsprintf": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "vsnprintf": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_U32, NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "puts": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "fputs": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "putchar": (NM_RET_I32, [NM_ARG_I32], 0, False),
    "fputc": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR], 0, False),
    "fwrite": (NM_RET_U32, [NM_ARG_PTR, NM_ARG_U32, NM_ARG_U32, NM_ARG_PTR], 0, False),
    "fread": (NM_RET_U32, [NM_ARG_PTR, NM_ARG_U32, NM_ARG_U32, NM_ARG_PTR], 0, False),
    "fopen": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "fopen64": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "fclose": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "fflush": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "fseek": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_I32, NM_ARG_I32], 0, False),
    "ftell": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "fgets": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_I32, NM_ARG_PTR], 0, False),
    "perror": (NM_RET_VOID, [NM_ARG_PTR], 0, False),
    "setvbuf": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_I32, NM_ARG_U32], 0, False),
    # File / POSIX
    "open": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_I32, NM_ARG_U32], 1, False),
    "open64": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_I32, NM_ARG_U32], 1, False),
    "openat": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR, NM_ARG_I32, NM_ARG_U32], 1, False),
    "close": (NM_RET_I32, [NM_ARG_I32], 0, False),
    "read": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR, NM_ARG_U32], 0, False),
    "write": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR, NM_ARG_U32], 0, False),
    "lseek": (NM_RET_I32, [NM_ARG_I32, NM_ARG_I32, NM_ARG_I32], 0, False),
    "lseek64": (NM_RET_I64, [NM_ARG_I32, NM_ARG_I64, NM_ARG_I32], 0, False),
    "unlink": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "symlink": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "readlink": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_U32], 0, False),
    "access": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_I32], 0, False),
    "fcntl": (NM_RET_I32, [NM_ARG_I32, NM_ARG_I32, NM_ARG_U32], 1, False),
    "ioctl": (NM_RET_I32, [NM_ARG_I32, NM_ARG_U32, NM_ARG_U32], 1, False),
    "dup": (NM_RET_I32, [NM_ARG_I32], 0, False),
    "dup2": (NM_RET_I32, [NM_ARG_I32, NM_ARG_I32], 0, False),
    "dup3": (NM_RET_I32, [NM_ARG_I32, NM_ARG_I32, NM_ARG_I32], 0, False),
    "pipe": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "pipe2": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_I32], 0, False),
    "isatty": (NM_RET_I32, [NM_ARG_I32], 0, False),
    "getpid": (NM_RET_I32, [], 0, False),
    "gettid": (NM_RET_I32, [], 0, False),
    "getuid": (NM_RET_U32, [], 0, False),
    "geteuid": (NM_RET_U32, [], 0, False),
    "getgid": (NM_RET_U32, [], 0, False),
    "getegid": (NM_RET_U32, [], 0, False),
    "sleep": (NM_RET_U32, [NM_ARG_U32], 0, False),
    "usleep": (NM_RET_I32, [NM_ARG_U32], 0, False),
    "mmap": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_U32, NM_ARG_I32, NM_ARG_I32, NM_ARG_I32, NM_ARG_I32], 0, False),
    "mmap64": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_U32, NM_ARG_I32, NM_ARG_I32, NM_ARG_I32, NM_ARG_I64], 0, False),
    "munmap": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_U32], 0, False),
    "mprotect": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_U32, NM_ARG_I32], 0, False),
    "uname": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "sysconf": (NM_RET_I32, [NM_ARG_I32], 0, False),
    # Struct repacking
    "stat": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "lstat": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "fstat": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR], 0, False),
    "fstatat": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR, NM_ARG_PTR, NM_ARG_I32], 0, False),
    "stat64": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "lstat64": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "fstat64": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR], 0, False),
    "fstatat64": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR, NM_ARG_PTR, NM_ARG_I32], 0, False),
    "__xstat": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "__lxstat": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "__fxstat": (NM_RET_I32, [NM_ARG_I32, NM_ARG_I32, NM_ARG_PTR], 0, False),
    "__fxstatat": (NM_RET_I32, [NM_ARG_I32, NM_ARG_I32, NM_ARG_PTR, NM_ARG_PTR, NM_ARG_I32], 0, False),
    "__xstat64": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "__lxstat64": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "__fxstat64": (NM_RET_I32, [NM_ARG_I32, NM_ARG_I32, NM_ARG_PTR], 0, False),
    "__fxstatat64": (NM_RET_I32, [NM_ARG_I32, NM_ARG_I32, NM_ARG_PTR, NM_ARG_PTR, NM_ARG_I32], 0, False),
    "__stat64_time64": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "__lstat64_time64": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "__fstat64_time64": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR], 0, False),
    "__fstatat64_time64": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR, NM_ARG_PTR, NM_ARG_I32], 0, False),
    "clock_gettime": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR], 0, False),
    "__clock_gettime64": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR], 0, False),
    "nanosleep": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "__nanosleep64": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "clock_nanosleep": (NM_RET_I32, [NM_ARG_I32, NM_ARG_I32, NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "__clock_nanosleep64": (NM_RET_I32, [NM_ARG_I32, NM_ARG_I32, NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "gettimeofday": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "time": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "getrlimit": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR], 0, False),
    "setrlimit": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR], 0, False),
    "getrlimit64": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR], 0, False),
    "setrlimit64": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR], 0, False),
    "prlimit": (NM_RET_I32, [NM_ARG_I32, NM_ARG_I32, NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "prlimit64": (NM_RET_I32, [NM_ARG_I32, NM_ARG_I32, NM_ARG_PTR, NM_ARG_PTR], 0, False),
    # Callbacks / pthreads / lifecycle
    "qsort": (NM_RET_VOID, [NM_ARG_PTR, NM_ARG_U32, NM_ARG_U32, NM_ARG_U32], 0, False),
    "bsearch": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_U32, NM_ARG_U32, NM_ARG_U32], 0, False),
    "pthread_create": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_U32, NM_ARG_U32], 0, False),
    "pthread_join": (NM_RET_I32, [NM_ARG_U32, NM_ARG_PTR], 0, False),
    "pthread_detach": (NM_RET_I32, [NM_ARG_U32], 0, False),
    "pthread_self": (NM_RET_U32, [], 0, False),
    "pthread_exit": (NM_RET_VOID, [NM_ARG_PTR], 0, False),
    "pthread_mutex_init": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "pthread_mutex_destroy": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "pthread_mutex_lock": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "pthread_mutex_trylock": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "pthread_mutex_unlock": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "pthread_cond_init": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "pthread_cond_destroy": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "pthread_cond_wait": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "pthread_cond_timedwait": (NM_RET_I32, [NM_ARG_PTR, NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "pthread_cond_signal": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "pthread_cond_broadcast": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    "atexit": (NM_RET_I32, [NM_ARG_U32], 0, False),
    "__cxa_atexit": (NM_RET_I32, [NM_ARG_U32, NM_ARG_U32, NM_ARG_U32], 0, False),
    "__cxa_finalize": (NM_RET_VOID, [NM_ARG_U32], 0, False),
    "signal": (NM_RET_U32, [NM_ARG_I32, NM_ARG_U32], 0, False),
    "sigaction": (NM_RET_I32, [NM_ARG_I32, NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "exit": (NM_RET_VOID, [NM_ARG_I32], 0, False),
    "_exit": (NM_RET_VOID, [NM_ARG_I32], 0, False),
    "abort": (NM_RET_VOID, [], 0, False),
    "__stack_chk_fail": (NM_RET_VOID, [], 0, False),
    "__errno_location": (NM_RET_PTR, [], 0, False),
    "__libc_start_main": (NM_RET_I32, [NM_ARG_U32, NM_ARG_I32, NM_ARG_PTR, NM_ARG_U32, NM_ARG_U32, NM_ARG_U32], 0, False),
    # dlfcn
    "dlopen": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_I32], 0, False),
    "dlsym": (NM_RET_PTR, [NM_ARG_PTR, NM_ARG_PTR], 0, False),
    "dlerror": (NM_RET_PTR, [], 0, False),
    "dlclose": (NM_RET_I32, [NM_ARG_PTR], 0, False),
    # libm
    "sqrt": (NM_RET_F64, [NM_ARG_F64], 0, False),
    "sqrtf": (NM_RET_F32, [NM_ARG_F32], 0, False),
    "sin": (NM_RET_F64, [NM_ARG_F64], 0, False),
    "sinf": (NM_RET_F32, [NM_ARG_F32], 0, False),
    "cos": (NM_RET_F64, [NM_ARG_F64], 0, False),
    "cosf": (NM_RET_F32, [NM_ARG_F32], 0, False),
    "tan": (NM_RET_F64, [NM_ARG_F64], 0, False),
    "tanf": (NM_RET_F32, [NM_ARG_F32], 0, False),
    "atan2": (NM_RET_F64, [NM_ARG_F64, NM_ARG_F64], 0, False),
    "pow": (NM_RET_F64, [NM_ARG_F64, NM_ARG_F64], 0, False),
    "powf": (NM_RET_F32, [NM_ARG_F32, NM_ARG_F32], 0, False),
    "exp": (NM_RET_F64, [NM_ARG_F64], 0, False),
    "log": (NM_RET_F64, [NM_ARG_F64], 0, False),
    "log10": (NM_RET_F64, [NM_ARG_F64], 0, False),
    "fabs": (NM_RET_F64, [NM_ARG_F64], 0, False),
    "fabsf": (NM_RET_F32, [NM_ARG_F32], 0, False),
    "floor": (NM_RET_F64, [NM_ARG_F64], 0, False),
    "ceil": (NM_RET_F64, [NM_ARG_F64], 0, False),
    "round": (NM_RET_F64, [NM_ARG_F64], 0, False),
}

COMPAT_DATA_SYMS = {
    "stdin",
    "stdout",
    "stderr",
    "optarg",
    "optind",
    "opterr",
    "optopt",
    "environ",
}

WEAK_CRT_SYMS = {
    "__gmon_start__",
    "_ITM_registerTMCloneTable",
    "_ITM_deregisterTMCloneTable",
}


def map_import_symbol_name(orig_name: str) -> str:
    """Map an i386 imported symbol name to its libnewminhh_compat.so wrapper symbol."""
    if orig_name in WEAK_CRT_SYMS:
        return orig_name
    if orig_name == "dlsym":
        return "compat_dlsym_cdecl"
    if orig_name in ("printf", "fprintf", "sprintf", "snprintf"):
        return f"compat_{orig_name}_cdecl"
    if orig_name in COMPAT_SIGNATURES or orig_name in COMPAT_DATA_SYMS:
        return f"compat_{orig_name}"
    return orig_name


def encode_nm_fn_sig(sym_name: str) -> bytes:
    """Encode nm_fn_sig_t (20 bytes) matching src/common/newminhh_abi.h."""
    if sym_name in COMPAT_SIGNATURES:
        ret_k, args, is_var, _ = COMPAT_SIGNATURES[sym_name]
    elif sym_name in WEAK_CRT_SYMS:
        ret_k, args, is_var = NM_RET_VOID, [], 0
    else:
        ret_k, args, is_var = NM_RET_U32, [NM_ARG_U32] * 6, 0
    padded = (args + [0] * 16)[:16]
    return struct.pack("<BBBB16B", ret_k, len(args), is_var, 0, *padded)


def assemble_and_link_helpers(
    asm_source: str,
    text_va: int,
    rodata_va: int,
    data_va: int,
    bss_va: int,
    defsyms: Dict[str, int],
) -> Tuple[bytes, bytes, bytes, int, Dict[str, int]]:
    """Assemble x32 GAS source and link at fixed section VAs using ld -m elf32_x86_64.

    Returns (text_bytes, rodata_bytes, data_bytes, bss_size, symbol_vas).
    """
    as_bin = shutil.which("as")
    ld_bin = shutil.which("ld")
    if not as_bin or not ld_bin:
        raise PatcherError("GNU 'as' and 'ld' are required inside re-env:latest")

    with tempfile.TemporaryDirectory(prefix="nm_asm_") as tmpdir:
        s_path = os.path.join(tmpdir, "helpers.s")
        o_path = os.path.join(tmpdir, "helpers.o")
        elf_path = os.path.join(tmpdir, "helpers.elf")
        lds_path = os.path.join(tmpdir, "helpers.lds")

        with open(s_path, "w", encoding="utf-8") as sf:
            sf.write(asm_source)

        with open(lds_path, "w", encoding="utf-8") as lf:
            lf.write(
                f"""SECTIONS
{{
  . = 0x{text_va:x};
  .text : {{ *(.text*) }}
  . = 0x{rodata_va:x};
  .rodata : {{ *(.rodata*) }}
  . = 0x{data_va:x};
  .data : {{ *(.data*) }}
  . = 0x{bss_va:x};
  .bss : {{ *(.bss*) *(COMMON) }}
}}
"""
            )

        p_as = subprocess.run([as_bin, "--x32", s_path, "-o", o_path], capture_output=True, text=True)
        if p_as.returncode != 0:
            raise PatcherError(f"GAS --x32 assembly failed:\n{p_as.stderr}")

        ld_cmd = [ld_bin, "-m", "elf32_x86_64", "-T", lds_path, "-o", elf_path]
        for k, v in defsyms.items():
            ld_cmd.extend(["--defsym", f"{k}=0x{v:x}"])
        ld_cmd.append(o_path)

        p_ld = subprocess.run(ld_cmd, capture_output=True, text=True)
        if p_ld.returncode != 0:
            raise PatcherError(f"ld -m elf32_x86_64 failed:\n{p_ld.stderr}")

        with open(elf_path, "rb") as ef:
            e = ELFFile(ef)
            tsec = e.get_section_by_name(".text")
            rsec = e.get_section_by_name(".rodata")
            dsec = e.get_section_by_name(".data")
            bsec = e.get_section_by_name(".bss")
            text_bytes = tsec.data() if tsec is not None else b""
            rodata_bytes = rsec.data() if rsec is not None else b""
            data_bytes = dsec.data() if dsec is not None else b""
            bss_size = bsec["sh_size"] if bsec is not None else 0

            sym_vas: Dict[str, int] = {}
            symtab = e.get_section_by_name(".symtab")
            if symtab is not None:
                for sym in symtab.iter_symbols():
                    if sym.name:
                        sym_vas[sym.name] = int(sym["st_value"])

    return text_bytes, rodata_bytes, data_bytes, bss_size, sym_vas


def generate_helper_assembly(
    plt_entries: List[Tuple[str, int, int]],  # (orig_sym_name, got_slot_va, dispatch_slot_idx)
    x32_entry_targets: List[Tuple[str, int]], # (label_name, rewritten_func_va)
    dual_abi_targets: List[Tuple[str, int]],  # (label_name, rewritten_func_va)
    fallback_funcs: List[Tuple[str, int]],    # (label_name, orig_func_va)
    orig_to_new_pairs: List[Tuple[int, int]], # (orig_va, new_va)
    copy_stdio_slots: List[Tuple[str, int]],  # (sym_name, dst_va)
    tls_info: Optional[Tuple[int, int, int, int, int]],
    orig_init_va: int,
    orig_entry_va: int,
    is_executable: bool,
    e_type: int,
    use_dbt_embedded_entry: bool,
    orig_elf_va: int,
    orig_elf_size: int,
    ref_va_for_bias: int,
    init_array_labels: Optional[List[str]] = None,
    fini_array_labels: Optional[List[str]] = None,
) -> str:
    """Generate complete x32 GAS assembly for runtime helpers, PLT stubs, and entry shims."""
    lines: List[str] = []
    lines.append(".text")

    # 1. Dispatch vector (8 bytes per slot at start of helpers_va):
    #    slot 0: jmp __nm_icall
    #    slot 1: jmp __nm_syscall_tramp
    #    slot 2: jmp __nm_ijmp
    #    slot 3..: jmp __nm_plt_<i>
    lines.append(".p2align 3")
    lines.append(".globl __nm_dispatch_vec")
    lines.append("__nm_dispatch_vec:")
    lines.append("    jmp __nm_icall")
    lines.append("    .p2align 3")
    lines.append("    jmp __nm_syscall_tramp")
    lines.append("    .p2align 3")
    lines.append("    jmp __nm_ijmp")
    lines.append("    .p2align 3")
    for idx, (_sname, _got_va, _slot_idx) in enumerate(plt_entries):
        lines.append(f"    jmp __nm_plt_{idx}")
        lines.append("    .p2align 3")

    # 2. __nm_ensure_gs_init (idempotent GS + PT_TLS + R_386_COPY stdio initializer)
    # Preserves all GPRs & RDX (_dl_fini)!
    lines.append(".globl __nm_ensure_gs_init")
    lines.append("__nm_ensure_gs_init:")
    lines.append("    pushq %rax")
    lines.append("    pushq %rcx")
    lines.append("    pushq %rdx")
    lines.append("    pushq %rsi")
    lines.append("    pushq %rdi")
    lines.append("    pushq %r8")
    lines.append("    pushq %r9")
    lines.append("    pushq %r10")
    lines.append("    pushq %r11")
    lines.append("    movq %rsp, %r10")
    lines.append("    andq $-16, %rsp")
    lines.append("    subq $32, %rsp")
    lines.append("    movq %r10, 16(%rsp)")
    lines.append("    movl $0, (%rsp)")
    lines.append("    movl $0x4000009e, %eax")  # __NR_arch_prctl | __X32_SYSCALL_BIT
    lines.append("    movl $0x1004, %edi")      # ARCH_GET_GS
    lines.append("    movl %esp, %esi")
    lines.append("    syscall")
    lines.append("    testl %eax, %eax")
    lines.append("    jnz .Lnm_set_gs")
    lines.append("    movl (%rsp), %eax")
    lines.append("    testl %eax, %eax")
    lines.append("    jnz .Lnm_gs_done")
    lines.append(".Lnm_set_gs:")
    lines.append("    leal __nm_tls_buf(%rip), %edx")
    lines.append("    addl $2048, %edx")        # %edx = TCB pointer
    if tls_info is not None and tls_info[2] > 0:
        tls_vaddr, _off, tls_filesz, tls_memsz, tls_align = tls_info
        tcb_alloc = ((max(tls_memsz, tls_filesz) + max(tls_align, 4) - 1) & ~(max(tls_align, 4) - 1))
        lines.append(f"    leal sym_orig_tls_data(%rip), %esi")
        lines.append(f"    movl %edx, %edi")
        lines.append(f"    subl ${tcb_alloc}, %edi")
        lines.append(f"    movl ${tls_filesz}, %ecx")
        lines.append("    cld")
        lines.append("    addr32 rep movsb")
    lines.append("    movl %edx, 0(%rdx)")      # gs:0 = tcb self ptr
    lines.append("    movl $0x54434231, 0x14(%rdx)")  # gs:0x14 = stack_chk_guard
    lines.append("    movl $0x4000009e, %eax")
    lines.append("    movl $0x1001, %edi")      # ARCH_SET_GS
    lines.append("    movl %edx, %esi")
    lines.append("    syscall")
    lines.append(".Lnm_gs_done:")
    if copy_stdio_slots:
        lines.append("    movl got_newminhh_get_stdio_streams(%rip), %r11d")
        lines.append("    testl %r11d, %r11d")
        lines.append("    jz .Lnm_stdio_done")
        lines.append("    leal 0(%rsp), %edi")
        lines.append("    leal 4(%rsp), %esi")
        lines.append("    leal 8(%rsp), %edx")
        lines.append("    call *%r11")
        for sname, dst_va in copy_stdio_slots:
            if sname == "stdin":
                src_off = 0
            elif sname == "stdout":
                src_off = 4
            elif sname == "stderr":
                src_off = 8
            else:
                continue
            lines.append(f"    movl {src_off}(%rsp), %eax")
            lines.append(f"    leal sym_copy_{sname}(%rip), %ecx")
            lines.append("    movl %eax, (%rcx)")
        lines.append(".Lnm_stdio_done:")
    lines.append("    movq 16(%rsp), %rsp")
    lines.append("    popq %r11")
    lines.append("    popq %r10")
    lines.append("    popq %r9")
    lines.append("    popq %r8")
    lines.append("    popq %rdi")
    lines.append("    popq %rsi")
    lines.append("    popq %rdx")
    lines.append("    popq %rcx")
    lines.append("    popq %rax")
    lines.append("    ret")

    # 3. _nm_init (DT_INIT entrypoint: x32 SysV, run-once guarded)
    lines.append(".globl _nm_init")
    lines.append("_nm_init:")
    lines.append("    cmpb $0, __nm_init_done(%rip)")
    lines.append("    jne .Lnm_init_ret")
    lines.append("    movb $1, __nm_init_done(%rip)")
    lines.append("    call __nm_ensure_gs_init")
    lines.append("    movl got_newminhh_register_native_cdecl_range(%rip), %r11d")
    lines.append("    testl %r11d, %r11d")
    lines.append("    jz .Lnm_init_after_reg")
    lines.append("    pushq %rdi")
    lines.append("    pushq %rsi")
    lines.append("    pushq %rdx")
    lines.append("    leal sym_text_new_start(%rip), %edi")
    lines.append("    leal sym_text_new_end(%rip), %esi")
    lines.append("    call *%r11")
    lines.append("    popq %rdx")
    lines.append("    popq %rsi")
    lines.append("    popq %rdi")
    lines.append(".Lnm_init_after_reg:")
    if orig_init_va != 0:
        lines.append("    leal sym_rewritten_init(%rip), %r11d")
        lines.append("    jmp __nm_x32_to_cdecl_common")
    else:
        lines.append(".Lnm_init_ret:")
        lines.append("    ret")
    if orig_init_va != 0:
        lines.append(".Lnm_init_ret:")
        lines.append("    ret")

    # 3b. _nm_fini (DT_FINI entrypoint: x32 SysV, unregisters native-cdecl range on dlclose/exit)
    lines.append(".globl _nm_fini")
    lines.append("_nm_fini:")
    lines.append("    cmpb $0, __nm_fini_done(%rip)")
    lines.append("    jne .Lnm_fini_ret")
    lines.append("    movb $1, __nm_fini_done(%rip)")
    lines.append("    movl got_newminhh_unregister_native_cdecl_range(%rip), %r11d")
    lines.append("    testl %r11d, %r11d")
    lines.append("    jz .Lnm_fini_ret")
    lines.append("    leal sym_text_new_start(%rip), %edi")
    lines.append("    leal sym_text_new_end(%rip), %esi")
    lines.append("    jmp *%r11")
    lines.append(".Lnm_fini_ret:")
    lines.append("    ret")

    # 4. _nm_start (e_entry for executables: x32 SysV)
    if is_executable:
        lines.append(".globl _nm_start")
        lines.append("_nm_start:")
        if use_dbt_embedded_entry:
            lines.append("    call __nm_ensure_gs_init")
            # _nm_start with %rsp = [argc, argv0, ..., NULL, envp0, ...]
            lines.append("    movl (%rsp), %ecx")        # argc
            lines.append("    leal 4(%rsp), %r8d")       # argv
            lines.append("    movl %ecx, %eax")
            lines.append("    addl $1, %eax")
            lines.append("    shll $2, %eax")
            lines.append("    leal (%r8, %rax, 1), %r9d")  # envp
            lines.append("    andq $-16, %rsp")
            lines.append("    subq $16, %rsp")
            lines.append("    leal sym_orig_elf_blob(%rip), %edi")
            lines.append(f"    movl ${orig_elf_size}, %esi")
            lines.append("    xorl %edx, %edx")
            lines.append("    movl got_newminhh_dbt_exec_embedded(%rip), %r11d")
            lines.append("    call *%r11")
            lines.append("    movl %eax, %edi")
            lines.append("    movl $0x400000e7, %eax")   # exit_group
            lines.append("    syscall")
        else:
            lines.append("    movq %rsp, %r12")
            lines.append("    movq %rdx, %r13")
            lines.append("    andq $-16, %rsp")
            lines.append("    movl (%r12), %edi")
            lines.append("    leal 4(%r12), %esi")
            lines.append("    leal 1(%edi), %eax")
            lines.append("    shll $2, %eax")
            lines.append("    leal (%rsi, %rax, 1), %edx")
            lines.append("    call _nm_init")
            for lbl in (fini_array_labels or []):
                lines.append(f"    leal target_{lbl}(%rip), %edi")
                lines.append("    movl got_compat_atexit(%rip), %r11d")
                lines.append("    testl %r11d, %r11d")
                lines.append(f"    jz .Lnm_skip_{lbl}")
                lines.append("    call *%r11")
                lines.append(f".Lnm_skip_{lbl}:")
            for lbl in (init_array_labels or []):
                lines.append("    movl (%r12), %edi")
                lines.append("    leal 4(%r12), %esi")
                lines.append("    leal 1(%edi), %eax")
                lines.append("    shll $2, %eax")
                lines.append("    leal (%rsi, %rax, 1), %edx")
                lines.append(f"    call {lbl}")
            lines.append("    movq %r13, %rdx")
            lines.append("    movq %r12, %rsp")
            lines.append("    xorl %r12d, %r12d")
            lines.append("    xorl %r13d, %r13d")
            lines.append("    xorl %eax, %eax")
            lines.append("    jmp sym_rewritten_entry")

    # 5. __nm_x32_to_cdecl_common (target cdecl VA in %r11d)
    lines.append(".globl __nm_x32_to_cdecl_common")
    lines.append("__nm_x32_to_cdecl_common:")
    lines.append("    pushq %rbx")
    lines.append("    pushq %rbp")
    lines.append("    pushq %r12")
    lines.append("    pushq %r13")
    lines.append("    pushq %r14")
    lines.append("    pushq %r15")
    lines.append("    movq %rsp, %r12")
    lines.append("    andq $-16, %rsp")
    lines.append("    subq $68, %rsp")
    lines.append("    movl %edi, 4(%rsp)")
    lines.append("    movl %esi, 8(%rsp)")
    lines.append("    movl %edx, 12(%rsp)")
    lines.append("    movl %ecx, 16(%rsp)")
    lines.append("    movl %r8d, 20(%rsp)")
    lines.append("    movl %r9d, 24(%rsp)")
    lines.append("    movl 56(%r12), %eax")
    lines.append("    movl %eax, 28(%rsp)")
    lines.append("    movl 64(%r12), %eax")
    lines.append("    movl %eax, 32(%rsp)")
    lines.append("    movl 72(%r12), %eax")
    lines.append("    movl %eax, 36(%rsp)")
    lines.append("    movl 80(%r12), %eax")
    lines.append("    movl %eax, 40(%rsp)")
    lines.append("    leal .Lnm_x32_ret(%rip), %eax")
    lines.append("    movl %eax, (%rsp)")
    lines.append("    jmp *%r11")
    lines.append(".Lnm_x32_ret:")
    lines.append("    movl %eax, %eax")
    lines.append("    movq %r12, %rsp")
    lines.append("    popq %r15")
    lines.append("    popq %r14")
    lines.append("    popq %r13")
    lines.append("    popq %r12")
    lines.append("    popq %rbp")
    lines.append("    popq %rbx")
    lines.append("    ret")

    # 6. Per-export / init_array / fini_array x32->cdecl stubs
    for lbl, _va in x32_entry_targets:
        lines.append(f".globl {lbl}")
        lines.append(f"{lbl}:")
        lines.append(f"    leal target_{lbl}(%rip), %r11d")
        lines.append("    jmp __nm_x32_to_cdecl_common")

    # 7. Dual-ABI common check + per-function dual-ABI stubs
    #    Entered with %r11d = target cdecl VA. Checks if (%rsp) is inside
    #    local [.text.newminhh], [newminhh_invoke_cdecl_asm .. +0x200),
    #    or any registered cross-DSO .text.newminhh range via newminhh_is_native_cdecl_va.
    lines.append("__nm_dual_abi_common:")
    lines.append("    movl (%rsp), %r10d")
    lines.append("    leal sym_text_new_start(%rip), %eax")
    lines.append("    cmpl %eax, %r10d")
    lines.append("    jb .Ldual_chk_compat")
    lines.append("    leal sym_text_new_end(%rip), %eax")
    lines.append("    cmpl %eax, %r10d")
    lines.append("    jb .Ldual_cdecl")
    lines.append(".Ldual_chk_compat:")
    lines.append("    movl got_newminhh_invoke_cdecl_asm(%rip), %eax")
    lines.append("    testl %eax, %eax")
    lines.append("    jz .Ldual_chk_reg")
    lines.append("    cmpl %eax, %r10d")
    lines.append("    jb .Ldual_chk_reg")
    lines.append("    addl $0x200, %eax")
    lines.append("    cmpl %eax, %r10d")
    lines.append("    jb .Ldual_cdecl")
    lines.append(".Ldual_chk_reg:")
    lines.append("    movl got_newminhh_is_native_cdecl_va(%rip), %eax")
    lines.append("    testl %eax, %eax")
    lines.append("    jz .Ldual_x32")
    lines.append("    movq %rsp, %r10")
    lines.append("    andq $-16, %rsp")
    lines.append("    subq $144, %rsp")
    lines.append("    movq %r10, 0(%rsp)")
    lines.append("    movq %r11, 8(%rsp)")
    lines.append("    movq %rdi, 16(%rsp)")
    lines.append("    movq %rsi, 24(%rsp)")
    lines.append("    movq %rdx, 32(%rsp)")
    lines.append("    movq %rcx, 40(%rsp)")
    lines.append("    movq %r8,  48(%rsp)")
    lines.append("    movq %r9,  56(%rsp)")
    lines.append("    movsd %xmm0,  64(%rsp)")
    lines.append("    movsd %xmm1,  72(%rsp)")
    lines.append("    movsd %xmm2,  80(%rsp)")
    lines.append("    movsd %xmm3,  88(%rsp)")
    lines.append("    movsd %xmm4,  96(%rsp)")
    lines.append("    movsd %xmm5, 104(%rsp)")
    lines.append("    movsd %xmm6, 112(%rsp)")
    lines.append("    movsd %xmm7, 120(%rsp)")
    lines.append("    movl (%r10), %edi")
    lines.append("    call *%rax")
    lines.append("    testl %eax, %eax")
    lines.append("    setnz %r10b")
    lines.append("    movsd 120(%rsp), %xmm7")
    lines.append("    movsd 112(%rsp), %xmm6")
    lines.append("    movsd 104(%rsp), %xmm5")
    lines.append("    movsd  96(%rsp), %xmm4")
    lines.append("    movsd  88(%rsp), %xmm3")
    lines.append("    movsd  80(%rsp), %xmm2")
    lines.append("    movsd  72(%rsp), %xmm1")
    lines.append("    movsd  64(%rsp), %xmm0")
    lines.append("    movq  56(%rsp), %r9")
    lines.append("    movq  48(%rsp), %r8")
    lines.append("    movq  40(%rsp), %rcx")
    lines.append("    movq  32(%rsp), %rdx")
    lines.append("    movq  24(%rsp), %rsi")
    lines.append("    movq  16(%rsp), %rdi")
    lines.append("    movq   8(%rsp), %r11")
    lines.append("    movq   0(%rsp), %rsp")
    lines.append("    testb %r10b, %r10b")
    lines.append("    jnz .Ldual_cdecl")
    lines.append(".Ldual_x32:")
    lines.append("    jmp __nm_x32_to_cdecl_common")
    lines.append(".Ldual_cdecl:")
    lines.append("    jmp *%r11")

    for lbl, _va in dual_abi_targets:
        lines.append(f".globl {lbl}")
        lines.append(f"{lbl}:")
        lines.append(f"    leal target_{lbl}(%rip), %r11d")
        lines.append("    jmp __nm_dual_abi_common")

    # 8. __nm_icall (indirect cdecl call to %r11d)
    lines.append(".globl __nm_icall")
    lines.append("__nm_icall:")
    lines.append("    leal sym_text_new_start(%rip), %r10d")
    lines.append("    cmpl %r10d, %r11d")
    lines.append("    jb .Lic_chk_map")
    lines.append("    leal sym_text_new_end(%rip), %r10d")
    lines.append("    cmpl %r10d, %r11d")
    lines.append("    jb .Lic_direct_cdecl")
    lines.append(".Lic_chk_map:")
    if orig_to_new_pairs:
        # Compute load_bias = __nm_dispatch_vec(%rip) - ref_va_for_bias
        lines.append("    leal __nm_dispatch_vec(%rip), %r10d")
        lines.append(f"    subl $0x{ref_va_for_bias:x}, %r10d")  # %r10d = load_bias (0 for ET_EXEC)
        lines.append("    movl %r11d, %eax")
        lines.append("    subl %r10d, %eax")                      # %eax = unbased VA
        lines.append("    leal __nm_orig_to_new_tbl(%rip), %ecx")
        lines.append(f"    movl ${len(orig_to_new_pairs)}, %edx")
        lines.append(".Lic_map_loop:")
        lines.append("    testl %edx, %edx")
        lines.append("    jz .Lic_chk_out_tramp")
        lines.append("    cmpl (%rcx), %eax")
        lines.append("    je .Lic_map_hit")
        lines.append("    addl $8, %ecx")
        lines.append("    decl %edx")
        lines.append("    jmp .Lic_map_loop")
        lines.append(".Lic_map_hit:")
        lines.append("    movl 4(%rcx), %r11d")
        lines.append("    addl %r10d, %r11d")
        lines.append("    jmp *%r11")
    lines.append(".Lic_chk_out_tramp:")
    lines.append("    movl got_newminhh_out_common(%rip), %r10d")
    lines.append("    testl %r10d, %r10d")
    lines.append("    jz .Lic_external_x32")
    lines.append("    cmpw $0xbb41, (%r11)")
    lines.append("    jne .Lic_external_x32")
    lines.append("    cmpb $0xe9, 6(%r11)")
    lines.append("    jne .Lic_external_x32")
    lines.append("    movslq 7(%r11), %rax")
    lines.append("    leal 11(%r11, %rax), %eax")
    lines.append("    cmpl %r10d, %eax")
    lines.append("    je .Lic_out_tramp")
    lines.append(".Lic_external_x32:")
    lines.append("    movq %rsp, %r10")
    lines.append("    andq $-16, %rsp")
    lines.append("    subq $32, %rsp")
    lines.append("    movq %r10, 0(%rsp)")
    lines.append("    movq %r12, 8(%rsp)")
    lines.append("    movq %rsi, 16(%rsp)")
    lines.append("    movq %rdi, 24(%rsp)")
    lines.append("    movq %r10, %r12")
    lines.append("    movl 4(%r12), %edi")
    lines.append("    movl 8(%r12), %esi")
    lines.append("    movl 12(%r12), %edx")
    lines.append("    movl 16(%r12), %ecx")
    lines.append("    movl 20(%r12), %r8d")
    lines.append("    movl 24(%r12), %r9d")
    lines.append("    xorl %eax, %eax")
    lines.append("    call *%r11")
    lines.append("    movq 24(%rsp), %rdi")
    lines.append("    movq 16(%rsp), %rsi")
    lines.append("    movq 8(%rsp), %r12")
    lines.append("    movq 0(%rsp), %rsp")
    lines.append("    movl (%rsp), %r11d")
    lines.append("    leaq 4(%rsp), %rsp")
    lines.append("    jmp *%r11")
    lines.append(".Lic_direct_cdecl:")
    lines.append("    jmp *%r11")
    lines.append(".Lic_out_tramp:")
    lines.append("    movq %rsp, %r10")
    lines.append("    andq $-16, %rsp")
    lines.append("    subq $112, %rsp")
    lines.append("    movq %r10, 80(%rsp)")
    lines.append("    movq %rsi, 88(%rsp)")
    lines.append("    movq %rdi, 96(%rsp)")
    lines.append("    movq %r12, 104(%rsp)")
    lines.append("    movq %rsp, %r12")
    lines.append("    xorl %ecx, %ecx")
    lines.append(".Lic_out_cp:")
    lines.append("    movl 4(%r10, %rcx, 4), %eax")
    lines.append("    movl %eax, 4(%r12, %rcx, 4)")
    lines.append("    incl %ecx")
    lines.append("    cmpl $16, %ecx")
    lines.append("    jb .Lic_out_cp")
    lines.append("    leal .Lic_out_ret(%rip), %eax")
    lines.append("    movl %eax, 0(%r12)")
    lines.append("    movq %r12, %rsp")
    lines.append("    jmp *%r11")
    lines.append(".Lic_out_ret:")
    lines.append("    movq 96(%r12), %rdi")
    lines.append("    movq 88(%r12), %rsi")
    lines.append("    movq 80(%r12), %r10")
    lines.append("    movq 104(%r12), %r12")
    lines.append("    movq %r10, %rsp")
    lines.append("    movl (%rsp), %r11d")
    lines.append("    leaq 4(%rsp), %rsp")
    lines.append("    jmp *%r11")

    # 8b. __nm_ijmp (tail indirect jump to %r11d without pushing a return address)
    lines.append(".globl __nm_ijmp")
    lines.append("__nm_ijmp:")
    lines.append("    leal sym_text_new_start(%rip), %r10d")
    lines.append("    cmpl %r10d, %r11d")
    lines.append("    jb .Lij_chk_map")
    lines.append("    leal sym_text_new_end(%rip), %r10d")
    lines.append("    cmpl %r10d, %r11d")
    lines.append("    jb .Lij_direct_cdecl")
    lines.append(".Lij_chk_map:")
    if orig_to_new_pairs:
        lines.append("    leal __nm_dispatch_vec(%rip), %r10d")
        lines.append(f"    subl $0x{ref_va_for_bias:x}, %r10d")
        lines.append("    movl %r11d, %eax")
        lines.append("    subl %r10d, %eax")
        lines.append("    leal __nm_orig_to_new_tbl(%rip), %ecx")
        lines.append(f"    movl ${len(orig_to_new_pairs)}, %edx")
        lines.append(".Lij_map_loop:")
        lines.append("    testl %edx, %edx")
        lines.append("    jz .Lij_chk_out_tramp")
        lines.append("    cmpl (%rcx), %eax")
        lines.append("    je .Lij_map_hit")
        lines.append("    addl $8, %ecx")
        lines.append("    decl %edx")
        lines.append("    jmp .Lij_map_loop")
        lines.append(".Lij_map_hit:")
        lines.append("    movl 4(%rcx), %r11d")
        lines.append("    addl %r10d, %r11d")
        lines.append("    jmp *%r11")
    lines.append(".Lij_chk_out_tramp:")
    lines.append("    movl got_newminhh_out_common(%rip), %r10d")
    lines.append("    testl %r10d, %r10d")
    lines.append("    jz .Lic_external_x32")
    lines.append("    cmpw $0xbb41, (%r11)")
    lines.append("    jne .Lic_external_x32")
    lines.append("    cmpb $0xe9, 6(%r11)")
    lines.append("    jne .Lic_external_x32")
    lines.append("    movslq 7(%r11), %rax")
    lines.append("    leal 11(%r11, %rax), %eax")
    lines.append("    cmpl %r10d, %eax")
    lines.append("    jne .Lic_external_x32")
    lines.append(".Lij_direct_cdecl:")
    lines.append("    jmp *%r11")

    # 9. __nm_syscall_tramp (int 0x80 / sysenter)
    lines.append(".globl __nm_syscall_tramp")
    lines.append("__nm_syscall_tramp:")
    lines.append("    movq %rsp, %r10")
    lines.append("    andq $-16, %rsp")
    lines.append("    subq $80, %rsp")
    lines.append("    movq %r10, 0(%rsp)")
    lines.append("    movq %rbx, 8(%rsp)")
    lines.append("    movq %rcx, 16(%rsp)")
    lines.append("    movq %rdx, 24(%rsp)")
    lines.append("    movq %rsi, 32(%rsp)")
    lines.append("    movq %rdi, 40(%rsp)")
    lines.append("    movq %rbp, 48(%rsp)")
    lines.append("    movl %eax, %edi")
    lines.append("    movl %ebx, %esi")
    lines.append("    movl 16(%rsp), %edx")
    lines.append("    movl 24(%rsp), %ecx")
    lines.append("    movl 32(%rsp), %r8d")
    lines.append("    movl 40(%rsp), %r9d")
    lines.append("    subq $16, %rsp")
    lines.append("    movl 64(%rsp), %eax")
    lines.append("    movq %rax, 0(%rsp)")
    lines.append("    movl got_newminhh_compat_syscall(%rip), %r11d")
    lines.append("    call *%r11")
    lines.append("    addq $16, %rsp")
    lines.append("    movq 48(%rsp), %rbp")
    lines.append("    movq 40(%rsp), %rdi")
    lines.append("    movq 32(%rsp), %rsi")
    lines.append("    movq 24(%rsp), %rdx")
    lines.append("    movq 16(%rsp), %rcx")
    lines.append("    movq 8(%rsp), %rbx")
    lines.append("    movq 0(%rsp), %rsp")
    lines.append("    movl (%rsp), %r11d")
    lines.append("    leaq 4(%rsp), %rsp")
    lines.append("    jmp *%r11")

    # 10. Per-PLT stubs (must preserve i386 callee-saved %rsi and %rdi!)
    for idx, (orig_sym, _got_va, _slot_idx) in enumerate(plt_entries):
        sig_info = COMPAT_SIGNATURES.get(orig_sym)
        is_cdecl_direct = sig_info[3] if sig_info else False
        lines.append(f".globl __nm_plt_{idx}")
        lines.append(f"__nm_plt_{idx}:")
        lines.append("    movq %rsp, %r10")
        lines.append("    andq $-16, %rsp")
        lines.append("    subq $32, %rsp")
        lines.append("    movq %r10, 0(%rsp)")
        lines.append("    movq %rsi, 8(%rsp)")
        lines.append("    movq %rdi, 16(%rsp)")
        lines.append(f"    movl sym_got_plt_{idx}(%rip), %edi")
        lines.append("    testl %edi, %edi")
        lines.append(f"    jz .Lplt_skip_{idx}")
        if is_cdecl_direct:
            lines.append("    movl %edi, %r11d")
            lines.append("    leaq 4(%r10), %rdi")
            lines.append("    call *%r11")
        else:
            lines.append(f"    leal __nm_sig_{idx}(%rip), %esi")
            lines.append("    leaq 4(%r10), %rdx")
            lines.append("    movl got_newminhh_call_x32_cdecl(%rip), %r11d")
            lines.append("    call *%r11")
        lines.append(f".Lplt_skip_{idx}:")
        lines.append("    movq 16(%rsp), %rdi")
        lines.append("    movq 8(%rsp), %rsi")
        lines.append("    movq 0(%rsp), %rsp")
        lines.append("    movl (%rsp), %r11d")
        lines.append("    leaq 4(%rsp), %rsp")
        lines.append("    jmp *%r11")

    # 11. Per-function hybrid DBT fallback stubs (newminhh_dbt_enter_at)
    for lbl, orig_va in fallback_funcs:
        lines.append(f".globl {lbl}")
        lines.append(f"{lbl}:")
        lines.append("    movq %rsp, %r10")
        lines.append("    andq $-16, %rsp")
        lines.append("    subq $768, %rsp")         # 592B newminhh_cpu_t + scratch stack (16B aligned)
        lines.append("    movq %r10, 608(%rsp)")
        lines.append("    movl %eax, 0(%rsp)")
        lines.append("    movl %ecx, 4(%rsp)")
        lines.append("    movl %edx, 8(%rsp)")
        lines.append("    movl %ebx, 12(%rsp)")
        lines.append("    movl %ebp, 20(%rsp)")
        lines.append("    movl %esi, 24(%rsp)")
        lines.append("    movl %edi, 28(%rsp)")
        # Zero newminhh_cpu_t offsets 32..591 (560 bytes = 70 qwords: eip, eflags, gs/fs/tls_bases, exit_reason, exit_arg, pad, fxsave_area)
        lines.append("    leal 32(%rsp), %edi")
        lines.append("    xorl %eax, %eax")
        lines.append("    movl $70, %ecx")
        lines.append("    cld")
        lines.append("    rep stosq")
        lines.append("    movl $0x202, 36(%rsp)")   # eflags
        # Copy 15 dwords from guest stack [r10+4..] to [rsp+644..704], sentinel at [rsp+640]
        lines.append("    leal 640(%rsp), %edi")
        lines.append("    movl $0xFFFF0000, (%rdi)")  # DBT_MAGIC_RET_TO_HOST
        lines.append("    movl $1, %ecx")
        lines.append(f".Lfb_cp_{lbl}:")
        lines.append("    movl (%r10, %rcx, 4), %eax")
        lines.append("    movl %eax, (%rdi, %rcx, 4)")
        lines.append("    incl %ecx")
        lines.append("    cmpl $16, %ecx")
        lines.append(f"    jb .Lfb_cp_{lbl}")
        lines.append("    movl %edi, 16(%rsp)")     # cpu.gpr[ESP]
        lines.append("    leal __nm_dispatch_vec(%rip), %edi")
        lines.append(f"    subl $0x{ref_va_for_bias:x}, %edi")
        lines.append(f"    addl $0x{orig_va:x}, %edi")
        lines.append("    movq %rsp, %rsi")
        lines.append("    movl got_newminhh_dbt_enter_at(%rip), %r11d")
        lines.append("    call *%r11")
        lines.append("    movl 0(%rsp), %eax")
        lines.append("    movl 8(%rsp), %edx")
        lines.append("    movl 12(%rsp), %ebx")
        lines.append("    movl 20(%rsp), %ebp")
        lines.append("    movl 24(%rsp), %esi")
        lines.append("    movl 28(%rsp), %edi")
        lines.append("    movq 608(%rsp), %rsp")
        lines.append("    movl (%rsp), %r11d")
        lines.append("    leaq 4(%rsp), %rsp")
        lines.append("    jmp *%r11")

    # .rodata section for signatures and orig_to_new table
    lines.append(".section .rodata")
    lines.append(".p2align 4")
    for idx, (orig_sym, _got_va, _slot_idx) in enumerate(plt_entries):
        sig_bytes = encode_nm_fn_sig(orig_sym)
        hex_bytes = ", ".join(f"0x{b:02x}" for b in sig_bytes)
        lines.append(f"__nm_sig_{idx}:")
        lines.append(f"    .byte {hex_bytes}")

    if orig_to_new_pairs:
        lines.append(".p2align 2")
        lines.append("__nm_orig_to_new_tbl:")
        for ova, nva in orig_to_new_pairs:
            lines.append(f"    .long 0x{ova:08x}, 0x{nva:08x}")

    # .bss section for main-thread GS TLS buffer (4 KiB) and init/fini guards
    lines.append(".bss")
    lines.append(".p2align 4")
    lines.append("__nm_tls_buf:")
    lines.append("    .skip 4096")
    lines.append("__nm_init_done:")
    lines.append("    .byte 0")
    lines.append("__nm_fini_done:")
    lines.append("    .byte 0")

    return "\n".join(lines) + "\n"
