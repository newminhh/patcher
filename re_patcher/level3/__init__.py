"""Level 3 decompiler-assisted re-patcher subpackage.

Provides pluggable decompiler backends (rz-ghidra, Ghidra headless, IDA Pro headless,
IDA Free assist-mode), static purity gates, C source sanitization, freestanding -mx32
boosting with two-tier differential verification, and Level 3C assist scaffolding.
"""

from __future__ import annotations

from .assist import AssistReport, generate_assist_scaffold
from .backends import (
    AVAILABLE_BACKENDS,
    BackendCapability,
    DecompiledFunction,
    DecompilerBackend,
    GhidraBackend,
    IDAFreeBackend,
    IDAProBackend,
    RzGhidraBackend,
    ingest_exported_dir,
    list_backend_capabilities,
    select_backend,
)
from .boost import (
    BoostedFunction,
    FunctionBoostOutcome,
    Level3BoostReport,
    compile_freestanding_x32,
    generate_cdecl_to_x32_boundary_stub,
    run_level3_boost,
)
from .diffverify import (
    DiffVerifyResult,
    can_run_i386_native,
    can_run_x32_native,
    generate_test_vectors,
    verify_boosted_function,
)
from .gates import (
    FunctionGateCandidate,
    GateVerdict,
    ScalarType,
    evaluate_purity_gate,
)
from .sanitize import (
    FunctionSignature,
    SanitizeResult,
    sanitize_function,
)

__all__ = [
    "AVAILABLE_BACKENDS",
    "AssistReport",
    "BackendCapability",
    "BoostedFunction",
    "DecompiledFunction",
    "DecompilerBackend",
    "DiffVerifyResult",
    "FunctionBoostOutcome",
    "FunctionGateCandidate",
    "FunctionSignature",
    "GateVerdict",
    "GhidraBackend",
    "IDAFreeBackend",
    "IDAProBackend",
    "Level3BoostReport",
    "RzGhidraBackend",
    "SanitizeResult",
    "ScalarType",
    "can_run_i386_native",
    "can_run_x32_native",
    "compile_freestanding_x32",
    "evaluate_purity_gate",
    "generate_assist_scaffold",
    "generate_cdecl_to_x32_boundary_stub",
    "generate_test_vectors",
    "ingest_exported_dir",
    "list_backend_capabilities",
    "run_level3_boost",
    "sanitize_function",
    "select_backend",
    "verify_boosted_function",
]
