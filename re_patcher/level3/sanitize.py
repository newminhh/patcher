"""C code sanitizer and strict safety checker for decompiled functions."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from .backends.base import DecompiledFunction
except (ImportError, ValueError):
    try:
        from backends.base import DecompiledFunction  # type: ignore
    except (ImportError, ValueError):
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from backends.base import DecompiledFunction  # type: ignore

FREESTANDING_HEADER = """#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Freestanding helper macros for Ghidra/Hex-Rays decompiler bit operations */
#ifndef CONCAT44
#define CONCAT44(hi, lo) (((uint64_t)(hi) << 32) | ((uint64_t)(uint32_t)(lo)))
#endif
#ifndef CONCAT22
#define CONCAT22(hi, lo) (((uint32_t)(hi) << 16) | ((uint32_t)(uint16_t)(lo)))
#endif
#ifndef CONCAT11
#define CONCAT11(hi, lo) (((uint16_t)(hi) << 8) | ((uint16_t)(uint8_t)(lo)))
#endif
#ifndef SUB84
#define SUB84(val, offset) ((uint32_t)(((uint64_t)(val)) >> ((offset) * 8)))
#endif
#ifndef SUB42
#define SUB42(val, offset) ((uint16_t)(((uint32_t)(val)) >> ((offset) * 8)))
#endif
#ifndef SUB41
#define SUB41(val, offset) ((uint8_t)(((uint32_t)(val)) >> ((offset) * 8)))
#endif
#ifndef SUB82
#define SUB82(val, offset) ((uint16_t)(((uint64_t)(val)) >> ((offset) * 8)))
#endif
#ifndef SUB81
#define SUB81(val, offset) ((uint8_t)(((uint64_t)(val)) >> ((offset) * 8)))
#endif
#ifndef ZEXT48
#define ZEXT48(val) ((uint64_t)(uint32_t)(val))
#endif
#ifndef SEXT48
#define SEXT48(val) ((int64_t)(int32_t)(val))
#endif
"""

SUPPORTED_MACROS = {
    "CONCAT44",
    "CONCAT22",
    "CONCAT11",
    "SUB84",
    "SUB42",
    "SUB41",
    "SUB82",
    "SUB81",
    "ZEXT48",
    "SEXT48",
}

TYPE_REPLACEMENTS = [
    (r"\bundefined1\b", "uint8_t"),
    (r"\bundefined2\b", "uint16_t"),
    (r"\bundefined4\b", "uint32_t"),
    (r"\bundefined8\b", "uint64_t"),
    (r"\bundefined\b", "uint8_t"),
    (r"\bbyte\b", "uint8_t"),
    (r"\bword\b", "uint16_t"),
    (r"\bdword\b", "uint32_t"),
    (r"\bqword\b", "uint64_t"),
    (r"\buint\b", "uint32_t"),
    (r"\bulong\b", "uint32_t"),
    (r"\blonglong\b", "int64_t"),
    (r"\bulonglong\b", "uint64_t"),
    (r"\bushort\b", "uint16_t"),
    (r"\buchar\b", "uint8_t"),
    (r"\bint1\b", "int8_t"),
    (r"\bint2\b", "int16_t"),
    (r"\bint4\b", "int32_t"),
    (r"\bint8\b", "int64_t"),
    (r"\buint1\b", "uint8_t"),
    (r"\buint2\b", "uint16_t"),
    (r"\buint4\b", "uint32_t"),
    (r"\buint8\b", "uint64_t"),
]

STRICT_FORBIDDEN_IDENTIFIERS_PATTERN = re.compile(
    r"\b("
    r"in_[A-Za-z0-9_]+"
    r"|unaff_[A-Za-z0-9_]+"
    r"|extraout_[A-Za-z0-9_]+"
    r"|in_stack_[0-9a-fA-F]+"
    r"|in_GS_OFFSET"
    r"|in_FS_OFFSET"
    r"|HALT"
    r"|LOCK"
    r"|SWI"
    r"|syscall"
    r"|asm"
    r"|__asm__"
    r"|auStack_[0-9a-fA-F_]*"
    r"|iStack_[0-9a-fA-F_]*"
    r"|uStack_[0-9a-fA-F_]*"
    r")\b"
)

DECOMPILER_WARNING_PATTERN = re.compile(
    r"(WARNING:|Bad instruction|Could not recover|overlapping|jumptable)",
    re.IGNORECASE,
)


@dataclass
class FunctionSignature:
    """Expected scalar signature for a function."""
    return_type: str
    param_types: List[str]


@dataclass
class SanitizeResult:
    """Outcome of decompiler C sanitization and strict safety verification."""
    ok: bool
    reason: Optional[str] = None
    c_code: Optional[str] = None
    sanitized_name: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def _strip_comments_and_strings(code: str) -> str:
    """Remove comments and string literals to simplify token and literal checking."""
    def replacer(match: re.Match) -> str:
        s = match.group(0)
        if s.startswith("/"):
            return " "  # replace comments with whitespace
        return '""'    # replace strings with empty string literal

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return pattern.sub(replacer, code)


def _extract_declared_identifiers(code_no_comments: str) -> Set[str]:
    """Extract declared local variable and parameter names from C source."""
    declared: Set[str] = set()

    # 1. Parameter names from function header
    func_match = re.search(
        r"\b[a-zA-Z_][a-zA-Z0-9_\*\s]+?\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*\{",
        code_no_comments,
    )
    if func_match:
        param_str = func_match.group(2).strip()
        if param_str and param_str != "void":
            for param in param_str.split(","):
                clean_param = re.sub(r"\[.*?\]", "", param).strip()
                tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", clean_param)
                if tokens:
                    declared.add(tokens[-1])

    # 2. Local variable declarations inside the function body
    decl_pattern = re.compile(
        r"(?:^|[;{}])\s*"
        r"(?:(?:const|volatile|static|register|unsigned|signed)\s+)*"
        r"(?:uint8_t|uint16_t|uint32_t|uint64_t|int8_t|int16_t|int32_t|int64_t|"
        r"int|uint|char|short|long|float|double|void|size_t|bool|undefined\d*|byte|word|dword|qword|[a-zA-Z_]\w*_t)"
        r"\s+([^;{}()]+);",
        re.MULTILINE,
    )

    for match in decl_pattern.finditer(code_no_comments):
        decl_body = match.group(1).strip()
        for part in decl_body.split(","):
            var_part = part.split("=")[0].strip()
            clean_var = re.sub(r"\[.*?\]", "", var_part)
            tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", clean_var)
            if tokens:
                declared.add(tokens[-1])

    return declared


def _check_integer_literals_in_segments(
    code_no_comments: str, segment_ranges: List[Tuple[int, int]]
) -> Optional[str]:
    """Check if any numeric literal falls into an analyzed ELF segment with start_va >= 0x1000.

    Allows small constants (< 0x1000) like math values 0xc, 0x100, 0xff.
    """
    eligible_segments = [(start, end) for start, end in segment_ranges if start >= 0x1000]
    if not eligible_segments:
        return None

    # Matches hex (0x1234) and decimal (1234) literals
    lit_pattern = re.compile(r"\b(0x[0-9a-fA-F]+|\d+)[uUlL]*\b")
    for match in lit_pattern.finditer(code_no_comments):
        token = match.group(1)
        try:
            val = int(token, 0)
        except ValueError:
            continue

        # Allow small constants (< 0x1000)
        if val < 0x1000:
            continue

        for seg_start, seg_end in eligible_segments:
            if seg_start <= val < seg_end:
                return (
                    f"Integer literal {hex(val)} falls inside ELF PT_LOAD segment "
                    f"[{hex(seg_start)}, {hex(seg_end)}) (constant-folded PIC pointer or absolute address)"
                )

    return None


def _check_signature_match(
    code_no_comments: str,
    func_name: str,
    expected_sig: FunctionSignature,
) -> Optional[str]:
    """Verify return type and parameter count against expected signature."""
    # Pattern to find function header
    pat = re.compile(
        r"([a-zA-Z_][a-zA-Z0-9_\*\s]+?)\b" + re.escape(func_name) + r"\s*\(([^)]*)\)\s*\{"
    )
    match = pat.search(code_no_comments)
    if not match:
        return None  # Cannot determine header, will be handled during renaming

    decomp_ret = match.group(1).strip()
    decomp_params_str = match.group(2).strip()

    # Determine decompiled parameter count
    if not decomp_params_str or decomp_params_str == "void":
        decomp_param_count = 0
    else:
        decomp_param_count = len([p for p in decomp_params_str.split(",") if p.strip()])

    expected_param_count = len(expected_sig.param_types)
    if decomp_param_count != expected_param_count:
        return (
            f"Parameter count mismatch: decompiled function has {decomp_param_count} parameters, "
            f"expected {expected_param_count}"
        )

    # Check return type voidness
    expected_is_void = expected_sig.return_type.strip() == "void"
    decomp_is_void = decomp_ret == "void"
    if expected_is_void != decomp_is_void:
        return (
            f"Return type mismatch: decompiled has '{decomp_ret}', "
            f"expected '{expected_sig.return_type}'"
        )

    return None


def sanitize_function(
    func: Union[DecompiledFunction, str],
    func_name: Optional[str] = None,
    expected_signature: Optional[Union[FunctionSignature, Tuple[str, List[str]], Dict[str, Any]]] = None,
    segment_ranges: Optional[List[Tuple[int, int]]] = None,
) -> SanitizeResult:
    """Sanitize and verify a decompiled function.

    Args:
        func: DecompiledFunction instance or raw C code string.
        func_name: Function symbol name (optional if func is DecompiledFunction).
        expected_signature: Verified signature from analyzer/gate.
        segment_ranges: List of (start_va, end_va) for ELF PT_LOAD segments.

    Returns:
        SanitizeResult indicating acceptance or strict safety rejection.
    """
    warnings: List[str] = []
    if isinstance(func, DecompiledFunction):
        raw_code = func.c_code
        name = func.name if func_name is None else func_name
        warnings.extend(func.warnings)
    else:
        raw_code = str(func)
        name = func_name or "unknown_func"

    if not raw_code.strip():
        return SanitizeResult(ok=False, reason="Empty decompiled C code", warnings=warnings)

    segment_ranges = segment_ranges or []

    # 1. STRICT REJECTION: Check decompiler warnings list & raw code warning comments
    for w in warnings:
        if DECOMPILER_WARNING_PATTERN.search(w):
            return SanitizeResult(
                ok=False,
                reason=f"Decompiler recorded failure warning: {w}",
                warnings=warnings,
            )

    warn_match = DECOMPILER_WARNING_PATTERN.search(raw_code)
    if warn_match:
        return SanitizeResult(
            ok=False,
            reason=f"Found decompiler warning/failure comment: '{warn_match.group(0)}'",
            warnings=warnings,
        )

    # 2. Convert decompiler typedefs to freestanding <stdint.h> types
    transformed_code = raw_code
    for pattern, repl in TYPE_REPLACEMENTS:
        transformed_code = re.sub(pattern, repl, transformed_code)

    # 3. Check for unhandled Ghidra bit-manipulation macros
    macro_matches = re.findall(r"\b(CONCAT\d+|SUB\d+|ZEXT\d+|SEXT\d+)\b", transformed_code)
    for m in macro_matches:
        if m not in SUPPORTED_MACROS:
            return SanitizeResult(
                ok=False,
                reason=f"Unhandled Ghidra macro '{m}'",
                warnings=warnings,
            )

    # Prepare code without comments and strings for token verification
    code_no_comments = _strip_comments_and_strings(transformed_code)

    # 4. STRICT REJECTION: Register-artifact identifiers and forbidden instructions
    forbidden_match = STRICT_FORBIDDEN_IDENTIFIERS_PATTERN.search(code_no_comments)
    if forbidden_match:
        return SanitizeResult(
            ok=False,
            reason=f"Found forbidden decompiler register artifact / instruction: '{forbidden_match.group(0)}'",
            warnings=warnings,
        )

    # Check bVar/uVar/iVar decompiler variable artifacts: allowed ONLY if declared
    declared_vars = _extract_declared_identifiers(code_no_comments)
    var_artifact_matches = set(re.findall(r"\b[biu]Var\d*\b", code_no_comments))
    for var in var_artifact_matches:
        if var not in declared_vars:
            return SanitizeResult(
                ok=False,
                reason=f"Undeclared decompiler variable artifact '{var}'",
                warnings=warnings,
            )

    # 5. STRICT REJECTION: Integer literal falling inside PT_LOAD segment >= 0x1000
    seg_err = _check_integer_literals_in_segments(code_no_comments, segment_ranges)
    if seg_err:
        return SanitizeResult(ok=False, reason=seg_err, warnings=warnings)

    # 6. STRICT REJECTION: Verified function signature mismatch
    if expected_signature is not None:
        if isinstance(expected_signature, tuple):
            sig_obj = FunctionSignature(return_type=expected_signature[0], param_types=expected_signature[1])
        elif isinstance(expected_signature, dict):
            sig_obj = FunctionSignature(
                return_type=expected_signature.get("return_type", "uint32_t"),
                param_types=expected_signature.get("param_types", []),
            )
        else:
            sig_obj = expected_signature

        sig_err = _check_signature_match(code_no_comments, name, sig_obj)
        if sig_err:
            return SanitizeResult(ok=False, reason=sig_err, warnings=warnings)

    # 7. Rename function symbol to nm_l3_<func_name>
    sanitized_name = f"nm_l3_{name}"
    rename_pattern = re.compile(r"(\b)" + re.escape(name) + r"(\s*\()")
    if not rename_pattern.search(transformed_code):
        # Fallback rename: rename first function definition
        transformed_code = re.sub(
            r"(\b[a-zA-Z_][a-zA-Z0-9_\*\s]+?\b)([a-zA-Z_][a-zA-Z0-9_]*)(\s*\([^)]*\)\s*\{)",
            rf"\1{sanitized_name}\3",
            transformed_code,
            count=1,
        )
    else:
        transformed_code = rename_pattern.sub(rf"\1{sanitized_name}\2", transformed_code, count=1)

    # 8. Wrap body with freestanding C23 headers and macros
    final_c_code = (
        f"{FREESTANDING_HEADER}\n"
        f"/* Sanitized Level 3 function: {sanitized_name} (orig: {name}) */\n\n"
        f"{transformed_code.strip()}\n"
    )

    return SanitizeResult(
        ok=True,
        reason=None,
        c_code=final_c_code,
        sanitized_name=sanitized_name,
        warnings=warnings,
    )


def _run_self_tests() -> bool:
    """Inline self-test verifying three required cases:

    1. plugin_compute -> ACCEPT
    2. plugin_name return 0x2000; -> REJECT
    3. switch_step in_ECX -> REJECT
    """
    print("[sanitize] Running inline self-tests...")
    segment_ranges = [(0x2000, 0x3000), (0x3000, 0x4000)]

    # Test 1: plugin_compute -> ACCEPT
    c1 = """
    undefined4 plugin_compute(undefined4 param_1, undefined4 param_2) {
        int iVar1;
        iVar1 = param_1 * 3;
        return iVar1 + param_2 + 0xc;
    }
    """
    sig1 = FunctionSignature(return_type="uint32_t", param_types=["uint32_t", "uint32_t"])
    res1 = sanitize_function(c1, "plugin_compute", sig1, segment_ranges)
    assert res1.ok, f"Test 1 failed: expected ACCEPT, got {res1.reason}"
    assert res1.sanitized_name == "nm_l3_plugin_compute"
    assert "nm_l3_plugin_compute" in (res1.c_code or "")
    assert "uint32_t" in (res1.c_code or "")
    print("  ✓ Test 1 (plugin_compute -> ACCEPT): PASSED")

    # Test 2: plugin_name return 0x2000; -> REJECT
    c2 = """
    char * plugin_name(void) {
        return (char *)0x2000;
    }
    """
    res2 = sanitize_function(c2, "plugin_name", None, segment_ranges)
    assert not res2.ok, f"Test 2 failed: expected REJECT, got ACCEPT"
    assert "falls inside ELF PT_LOAD segment" in (res2.reason or "")
    print(f"  ✓ Test 2 (plugin_name return 0x2000; -> REJECT): PASSED ({res2.reason})")

    # Test 3: switch_step in_ECX -> REJECT
    c3 = """
    int switch_step(int step) {
        if (in_ECX == 0) return 1;
        return step + 1;
    }
    """
    res3 = sanitize_function(c3, "switch_step", None, segment_ranges)
    assert not res3.ok, f"Test 3 failed: expected REJECT, got ACCEPT"
    assert "in_ECX" in (res3.reason or "")
    print(f"  ✓ Test 3 (switch_step in_ECX -> REJECT): PASSED ({res3.reason})")

    print("[sanitize] All inline self-tests passed successfully!")
    return True


if __name__ == "__main__":
    _run_self_tests()
