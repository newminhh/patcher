"""RzGhidra (rizin + rz-ghidra plugin) decompiler backend (headless, zero-JVM)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

from .base import BackendCapability, DecompiledFunction, DecompilerBackend


class RzGhidraBackend(DecompilerBackend):
    """Decompiler backend utilizing rizin with the rz-ghidra plugin."""

    @property
    def name(self) -> str:
        return "rz-ghidra"

    def probe(self) -> BackendCapability:
        """Probe for rizin executable and the rz-ghidra (core_ghidra) plugin."""
        rz_path = shutil.which("rizin")
        if not rz_path:
            return BackendCapability(
                name=self.name,
                available=False,
                headless=True,
                assist_only=False,
                path=None,
                reason="rizin executable not found in PATH",
            )

        try:
            # Check rizin core plugins for core_ghidra
            proc = subprocess.run(
                [rz_path, "-qc", "Lc", "--"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            if "core_ghidra" in out or "ghidra" in out.lower():
                return BackendCapability(
                    name=self.name,
                    available=True,
                    headless=True,
                    assist_only=False,
                    path=rz_path,
                    reason="rizin with rz-ghidra plugin detected",
                )

            # Fallback probe: check rizin -L
            proc2 = subprocess.run(
                [rz_path, "-L"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            out2 = (proc2.stdout or "") + (proc2.stderr or "")
            if "core_ghidra" in out2 or "ghidra" in out2.lower():
                return BackendCapability(
                    name=self.name,
                    available=True,
                    headless=True,
                    assist_only=False,
                    path=rz_path,
                    reason="rizin with rz-ghidra plugin detected",
                )

            return BackendCapability(
                name=self.name,
                available=False,
                headless=True,
                assist_only=False,
                path=rz_path,
                reason="rizin executable found, but rz-ghidra (core_ghidra) plugin is missing",
            )
        except Exception as exc:
            return BackendCapability(
                name=self.name,
                available=False,
                headless=True,
                assist_only=False,
                path=rz_path,
                reason=f"Failed probing rizin plugins: {exc}",
            )

    def decompile_batch(
        self, elf_path: str, funcs: List[Tuple[int, str]], timeout: int = 120
    ) -> Dict[int, DecompiledFunction]:
        """Decompile functions in a single rizin invocation so `aaa` runs once."""
        if not funcs:
            return {}

        rz_path = shutil.which("rizin")
        if not rz_path:
            raise RuntimeError("rizin executable not found in PATH")

        # Build chained command sequence:
        # aaa; s <va1>; echo ===NM_FUNC_START:<va1>===; pdg; s <va2>; ...
        cmd_parts: List[str] = ["aaa"]
        for va, _ in funcs:
            va_hex = hex(va)
            cmd_parts.append(f"s {va_hex}")
            cmd_parts.append(f"echo ===NM_FUNC_START:{va_hex}===")
            cmd_parts.append("pdg")

        cmd_string = "; ".join(cmd_parts)

        cmd = [
            rz_path,
            "-2",
            "-q",
            "-e",
            "scr.color=0",
            "-c",
            cmd_string,
            elf_path,
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        output = proc.stdout or ""
        return self._parse_marked_output(output, funcs)

    def _parse_marked_output(
        self, output: str, funcs: List[Tuple[int, str]]
    ) -> Dict[int, DecompiledFunction]:
        """Parse function blocks demarcated by ===NM_FUNC_START:<va>=== markers."""
        results: Dict[int, DecompiledFunction] = {}
        marker_pattern = re.compile(r"===NM_FUNC_START:(0x[0-9a-fA-F]+|\d+)===")

        matches = list(marker_pattern.finditer(output))
        funcs_by_va = {va: name for va, name in funcs}

        for i, match in enumerate(matches):
            raw_va_str = match.group(1)
            va = int(raw_va_str, 0)
            func_name = funcs_by_va.get(va, f"sub_{va:x}")

            start_idx = match.end()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(output)
            block = output[start_idx:end_idx].strip()

            c_code, prototype, warnings = self._extract_c_and_proto(block, func_name)
            results[va] = DecompiledFunction(
                name=func_name,
                va=va,
                c_code=c_code,
                prototype=prototype,
                backend=self.name,
                warnings=warnings,
            )

        # For any requested func missing from markers, record empty result with warning
        for va, name in funcs:
            if va not in results:
                results[va] = DecompiledFunction(
                    name=name,
                    va=va,
                    c_code="",
                    prototype="",
                    backend=self.name,
                    warnings=["No output returned by rizin for this function"],
                )

        return results

    @staticmethod
    def _extract_c_and_proto(block: str, default_name: str) -> Tuple[str, str, List[str]]:
        """Extract clean C code, prototype, and warnings from a rizin pdg block."""
        lines = block.splitlines()
        warnings: List[str] = []
        cleaned_lines: List[str] = []

        warning_regex = re.compile(
            r"(WARNING:|Bad instruction|Could not recover|overlapping|jumptable)",
            re.IGNORECASE,
        )

        for line in lines:
            if warning_regex.search(line):
                warnings.append(line.strip())
            cleaned_lines.append(line)

        c_code = "\n".join(cleaned_lines).strip()
        prototype = ""

        # Search for function declaration: line containing default_name or opening brace
        proto_pattern = re.compile(
            r"^[a-zA-Z_][a-zA-Z0-9_\s\*]+\b" + re.escape(default_name) + r"\s*\([^)]*\)",
            re.MULTILINE,
        )
        match = proto_pattern.search(c_code)
        if match:
            prototype = match.group(0).strip()
        else:
            # Fallback: scan for any function header before '{'
            for idx, line in enumerate(cleaned_lines):
                sline = line.strip()
                if "{" in sline:
                    prev = " ".join([l.strip() for l in cleaned_lines[max(0, idx - 2) : idx + 1]])
                    cand = prev.split("{")[0].strip()
                    if "(" in cand and ")" in cand:
                        prototype = cand
                    break

        return c_code, prototype, warnings
