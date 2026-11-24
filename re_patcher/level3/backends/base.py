"""Base classes and data models for pluggable decompiler backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DecompiledFunction:
    """Represents a decompiled C function emitted by a backend."""

    name: str
    va: int
    c_code: str
    prototype: str
    backend: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "va": self.va,
            "c_code": self.c_code,
            "prototype": self.prototype,
            "backend": self.backend,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DecompiledFunction:
        return cls(
            name=str(data["name"]),
            va=int(data["va"]),
            c_code=str(data.get("c_code", "")),
            prototype=str(data.get("prototype", "")),
            backend=str(data.get("backend", "unknown")),
            warnings=list(data.get("warnings", [])),
        )


@dataclass
class BackendCapability:
    """Represents capability and availability status of a decompiler backend."""

    name: str
    available: bool
    headless: bool
    assist_only: bool
    path: Optional[str]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "headless": self.headless,
            "assist_only": self.assist_only,
            "path": self.path,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BackendCapability:
        return cls(
            name=str(data["name"]),
            available=bool(data["available"]),
            headless=bool(data["headless"]),
            assist_only=bool(data.get("assist_only", False)),
            path=data.get("path"),
            reason=str(data.get("reason", "")),
        )


class DecompilerBackend(ABC):
    """Abstract base class for decompiler backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this backend (e.g. 'rz-ghidra', 'ghidra', 'idapro', 'idafree')."""
        ...

    @abstractmethod
    def probe(self) -> BackendCapability:
        """Probe environment, binaries, or plugins to check if this backend can operate."""
        ...

    def decompile_function(
        self, elf_path: str, func_va: int, func_name: str, timeout: int = 30
    ) -> DecompiledFunction:
        """Decompile a single function from an ELF binary."""
        batch_results = self.decompile_batch(
            elf_path=elf_path,
            funcs=[(func_va, func_name)],
            timeout=timeout,
        )
        if func_va in batch_results:
            return batch_results[func_va]
        raise RuntimeError(
            f"Backend '{self.name}' failed to decompile function '{func_name}' at 0x{func_va:x}"
        )

    @abstractmethod
    def decompile_batch(
        self, elf_path: str, funcs: List[Tuple[int, str]], timeout: int = 120
    ) -> Dict[int, DecompiledFunction]:
        """Decompile a batch of functions from an ELF binary.

        Args:
            elf_path: Path to the target ELF binary.
            funcs: List of (virtual_address, function_name) tuples to decompile.
            timeout: Subprocess timeout in seconds.

        Returns:
            Dictionary mapping function VA to DecompiledFunction.
        """
        ...
