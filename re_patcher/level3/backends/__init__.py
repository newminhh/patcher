"""Pluggable decompiler backends registry and selection for newminhh Level 3."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from .base import BackendCapability, DecompiledFunction, DecompilerBackend
from .ghidra import GhidraBackend
from .idafree import IDAFreeBackend, ingest_exported_dir
from .idapro import IDAProBackend
from .rzghidra import RzGhidraBackend

try:
    from ... import PatcherError
except ImportError:
    class PatcherError(Exception):  # type: ignore
        """Fallback exception if imported outside re_patcher package."""
        pass


AVAILABLE_BACKENDS: Dict[str, Type[DecompilerBackend]] = {
    "rz-ghidra": RzGhidraBackend,
    "ghidra": GhidraBackend,
    "idapro": IDAProBackend,
    "idafree": IDAFreeBackend,
}


def list_backend_capabilities() -> List[BackendCapability]:
    """Probe all registered decompiler backends and return their capabilities."""
    caps: List[BackendCapability] = []
    for name, backend_cls in AVAILABLE_BACKENDS.items():
        backend = backend_cls()
        caps.append(backend.probe())
    return caps


def select_backend(choice: str = "auto", require_headless: bool = True) -> DecompilerBackend:
    """Select and instantiate a decompiler backend.

    Args:
        choice: Backend identifier: 'auto', 'rz-ghidra', 'ghidra', 'idapro', 'idafree', or 'ida'.
        require_headless: If True, reject assist-only / non-headless backends (e.g. IDA Free).

    Returns:
        An instance of DecompilerBackend ready for decompilation.

    Raises:
        PatcherError: If no suitable or requested backend is available.
    """
    normalized = choice.strip().lower().replace("_", "-")

    # Map aliases
    if normalized == "rzghidra":
        normalized = "rz-ghidra"
    elif normalized in ("ida-pro", "idap"):
        normalized = "idapro"
    elif normalized in ("ida-free", "free"):
        normalized = "idafree"

    # Instantiate candidate backends
    backend_rz = RzGhidraBackend()
    backend_gh = GhidraBackend()
    backend_idapro = IDAProBackend()
    backend_idafree = IDAFreeBackend()

    if normalized == "auto":
        # Probe headless candidates in priority order: rz-ghidra -> ghidra -> idapro
        headless_candidates: List[DecompilerBackend] = [backend_rz, backend_gh, backend_idapro]
        for backend in headless_candidates:
            cap = backend.probe()
            if cap.available and cap.headless:
                return backend

        # If unattended/headless is NOT strictly required (assist mode), fallback to idafree
        if not require_headless:
            cap_free = backend_idafree.probe()
            if cap_free.available:
                return backend_idafree

        reasons = [f"{b.name}: {b.probe().reason}" for b in headless_candidates]
        msg = (
            "No headless decompiler backend is available on this system.\n"
            + "\n".join(f"  - {r}" for r in reasons)
        )
        if require_headless:
            msg += "\nTo use interactive IDA Free, enable assist mode (--decompile-assist)."
        raise PatcherError(msg)

    elif normalized == "ida":
        # Auto-pick IDA Pro if headless is available
        cap_pro = backend_idapro.probe()
        if cap_pro.available:
            return backend_idapro

        if not require_headless:
            cap_free = backend_idafree.probe()
            if cap_free.available:
                return backend_idafree
            raise PatcherError(
                f"Neither IDA Pro nor IDA Free could be detected.\n"
                f"IDA Pro: {cap_pro.reason}\n"
                f"IDA Free: {cap_free.reason}"
            )
        else:
            raise PatcherError(
                f"IDA Pro headless (idalib/idat) is required for unattended Level 3 mode, "
                f"but it is not available: {cap_pro.reason}. "
                f"For IDA Free, run with --decompile-assist."
            )

    elif normalized in ("rz-ghidra", "ghidra", "idapro", "idafree"):
        backend_map = {
            "rz-ghidra": backend_rz,
            "ghidra": backend_gh,
            "idapro": backend_idapro,
            "idafree": backend_idafree,
        }
        backend = backend_map[normalized]
        cap = backend.probe()

        if require_headless and not cap.headless:
            raise PatcherError(
                f"Requested backend '{normalized}' is not headless (assist_only={cap.assist_only}), "
                f"but require_headless=True was specified."
            )

        if not cap.available:
            raise PatcherError(
                f"Requested decompiler backend '{normalized}' is not available: {cap.reason}"
            )

        return backend

    else:
        valid_options = ["auto", "rz-ghidra", "ghidra", "ida", "idapro", "idafree"]
        raise PatcherError(
            f"Unknown decompiler backend choice '{choice}'. Valid choices: {', '.join(valid_options)}"
        )


__all__ = [
    "BackendCapability",
    "DecompiledFunction",
    "DecompilerBackend",
    "RzGhidraBackend",
    "GhidraBackend",
    "IDAProBackend",
    "IDAFreeBackend",
    "ingest_exported_dir",
    "list_backend_capabilities",
    "select_backend",
]
