"""Headless IDA Pro decompiler backend (idalib / idat)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

from .base import BackendCapability, DecompiledFunction, DecompilerBackend

IDALIB_RUNNER_SCRIPT = """# Helper runner for idalib subprocess
import sys
import os
import json

ida_dir = os.environ.get("IDADIR", "/opt/ida-pro")
for p in [os.path.join(ida_dir, "idalib", "python"), os.path.join(ida_dir, "python")]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    import idapro
    import ida_auto
    import ida_hexrays
    import ida_funcs
    import ida_lines
except ImportError as exc:
    sys.stderr.write(f"IDALIB_IMPORT_ERROR: {exc}\\n")
    sys.exit(2)

elf_path = sys.argv[1]
in_json = sys.argv[2]
out_json = sys.argv[3]

with open(in_json, "r", encoding="utf-8") as f:
    targets = json.load(f)

# Open database without prompting
try:
    db = idapro.open_database(elf_path, True)
except Exception:
    try:
        db = idapro.open_database(elf_path, run_auto=True)
    except Exception as exc:
        sys.stderr.write(f"OPEN_DB_ERROR: {exc}\\n")
        sys.exit(3)

results = {}
try:
    ida_auto.auto_wait()
    if not ida_hexrays.init_hexrays_plugin():
        sys.stderr.write("HEXRAYS_INIT_ERROR: Hex-Rays decompiler plugin failed to initialize\\n")
        sys.exit(4)

    for item in targets:
        va = int(item[0])
        name = str(item[1])
        cfunc = ida_hexrays.decompile(va)
        if cfunc:
            lines = []
            sv = cfunc.get_pseudocode()
            for i in range(len(sv)):
                lines.append(ida_lines.tag_remove(sv[i].line))
            c_code = "\\n".join(lines)
            proto = str(cfunc.type) if hasattr(cfunc, "type") else ""
            results[str(va)] = {
                "name": name,
                "va": va,
                "c_code": c_code,
                "prototype": proto,
                "backend": "idapro",
                "warnings": []
            }
        else:
            results[str(va)] = {
                "name": name,
                "va": va,
                "c_code": "",
                "prototype": "",
                "backend": "idapro",
                "warnings": ["Hex-Rays decompile returned None for function"]
            }
finally:
    try:
        if hasattr(idapro, "close_database"):
            idapro.close_database(False)
    except Exception:
        pass

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f)

sys.exit(0)
"""

IDAT_RUNNER_SCRIPT = """# Helper runner for idat batch mode
import sys
import os
import json
import idc
import ida_auto
import ida_hexrays
import ida_lines
import ida_funcs

in_json = os.environ.get("NM_IDAT_IN")
out_json = os.environ.get("NM_IDAT_OUT")

if not in_json or not out_json:
    args = idc.ARGV if hasattr(idc, "ARGV") else sys.argv
    if len(args) >= 3:
        in_json = args[1]
        out_json = args[2]

results = {}
try:
    ida_auto.auto_wait()
    if not ida_hexrays.init_hexrays_plugin():
        sys.stderr.write("HEXRAYS_INIT_ERROR: Hex-Rays plugin init failed\\n")
        idc.qexit(4)

    with open(in_json, "r", encoding="utf-8") as f:
        targets = json.load(f)

    for item in targets:
        va = int(item[0])
        name = str(item[1])
        cfunc = ida_hexrays.decompile(va)
        if cfunc:
            lines = []
            sv = cfunc.get_pseudocode()
            for i in range(len(sv)):
                lines.append(ida_lines.tag_remove(sv[i].line))
            c_code = "\\n".join(lines)
            proto = str(cfunc.type) if hasattr(cfunc, "type") else ""
            results[str(va)] = {
                "name": name,
                "va": va,
                "c_code": c_code,
                "prototype": proto,
                "backend": "idapro",
                "warnings": []
            }
        else:
            results[str(va)] = {
                "name": name,
                "va": va,
                "c_code": "",
                "prototype": "",
                "backend": "idapro",
                "warnings": ["Hex-Rays decompile returned None"]
            }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)
    idc.qexit(0)
except Exception as exc:
    sys.stderr.write(f"IDAT_ERROR: {exc}\\n")
    idc.qexit(1)
"""


class IDAProBackend(DecompilerBackend):
    """Headless IDA Pro decompiler backend using idalib or idat."""

    @property
    def name(self) -> str:
        return "idapro"

    def _find_ida_dir(self) -> Optional[str]:
        """Check IDADIR env var or default /opt/ida-pro."""
        env_dir = os.environ.get("IDADIR")
        if env_dir and os.path.isdir(env_dir):
            return env_dir
        if os.path.isdir("/opt/ida-pro"):
            return "/opt/ida-pro"
        return None

    def _find_idat_binary(self) -> Optional[str]:
        """Locate idat or idat64 executable."""
        ida_dir = self._find_ida_dir()
        if ida_dir:
            for cand in ["idat64", "idat"]:
                path = os.path.join(ida_dir, cand)
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    return path

        for cand in ["idat64", "idat"]:
            which = shutil.which(cand)
            if which and os.access(which, os.X_OK):
                return which
        return None

    def _has_idalib(self) -> bool:
        """Check if idalib/idapro is importable or present in IDA directory."""
        ida_dir = self._find_ida_dir()
        if ida_dir:
            cand = os.path.join(ida_dir, "idalib", "python")
            if os.path.isdir(cand):
                return True
        try:
            import idapro  # type: ignore # noqa: F401
            return True
        except ImportError:
            pass
        try:
            import idalib  # type: ignore # noqa: F401
            return True
        except ImportError:
            pass
        return False

    def probe(self) -> BackendCapability:
        """Probe for IDA Pro headless capabilities (idalib or idat)."""
        has_idalib = self._has_idalib()
        idat_bin = self._find_idat_binary()

        if has_idalib or idat_bin:
            primary = "idalib" if has_idalib else idat_bin
            return BackendCapability(
                name=self.name,
                available=True,
                headless=True,
                assist_only=False,
                path=primary,
                reason="IDA Pro headless interface detected (idalib/idat)",
            )

        return BackendCapability(
            name=self.name,
            available=False,
            headless=True,
            assist_only=False,
            path=None,
            reason="IDA Pro headless interface not found (checked IDADIR, /opt/ida-pro, idalib, idat)",
        )

    def decompile_batch(
        self, elf_path: str, funcs: List[Tuple[int, str]], timeout: int = 120
    ) -> Dict[int, DecompiledFunction]:
        """Decompile functions in a batch using idalib or idat fallback in a temp directory."""
        if not funcs:
            return {}

        with tempfile.TemporaryDirectory(prefix="nm_idapro_l3_") as tmpdir:
            # Copy target ELF into temp dir so .i64/.idb files are never created in workspace!
            tmp_elf = os.path.join(tmpdir, os.path.basename(elf_path))
            shutil.copy2(elf_path, tmp_elf)

            in_json = os.path.join(tmpdir, "targets.json")
            out_json = os.path.join(tmpdir, "idapro_out.json")
            with open(in_json, "w", encoding="utf-8") as f:
                json.dump([[va, name] for va, name in funcs], f)

            # Strategy 1: Attempt idalib via clean subprocess
            if self._has_idalib():
                res = self._run_idalib_subprocess(tmpdir, tmp_elf, in_json, out_json, timeout)
                if res is not None:
                    return self._build_results(res, funcs)

            # Strategy 2: Fallback to idat
            idat_bin = self._find_idat_binary()
            if idat_bin:
                res = self._run_idat_subprocess(idat_bin, tmpdir, tmp_elf, in_json, out_json, timeout)
                if res is not None:
                    return self._build_results(res, funcs)

            raise RuntimeError("IDA Pro headless decompilation failed (neither idalib nor idat succeeded)")

    def _run_idalib_subprocess(
        self, tmpdir: str, elf_path: str, in_json: str, out_json: str, timeout: int
    ) -> Optional[Dict[str, dict]]:
        script_file = os.path.join(tmpdir, "run_idalib.py")
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(IDALIB_RUNNER_SCRIPT)

        env = dict(os.environ)
        ida_dir = self._find_ida_dir()
        if ida_dir:
            env["IDADIR"] = ida_dir
            lib_py = os.path.join(ida_dir, "idalib", "python")
            if os.path.isdir(lib_py):
                existing_pythonpath = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f"{lib_py}:{existing_pythonpath}" if existing_pythonpath else lib_py

        proc = subprocess.run(
            [sys.executable, script_file, elf_path, in_json, out_json],
            cwd=tmpdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        if proc.returncode == 0 and os.path.isfile(out_json):
            try:
                with open(out_json, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _run_idat_subprocess(
        self, idat_bin: str, tmpdir: str, elf_path: str, in_json: str, out_json: str, timeout: int
    ) -> Optional[Dict[str, dict]]:
        script_file = os.path.join(tmpdir, "run_idat.py")
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(IDAT_RUNNER_SCRIPT)

        env = dict(os.environ)
        env["NM_IDAT_IN"] = in_json
        env["NM_IDAT_OUT"] = out_json

        cmd = [
            idat_bin,
            "-A",
            f"-S{script_file}",
            elf_path,
        ]

        proc = subprocess.run(
            cmd,
            cwd=tmpdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        if os.path.isfile(out_json):
            try:
                with open(out_json, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _build_results(
        self, raw_data: Dict[str, dict], funcs: List[Tuple[int, str]]
    ) -> Dict[int, DecompiledFunction]:
        results: Dict[int, DecompiledFunction] = {}
        for va, default_name in funcs:
            str_va = str(va)
            if str_va in raw_data:
                item = raw_data[str_va]
                results[va] = DecompiledFunction(
                    name=item.get("name", default_name),
                    va=va,
                    c_code=item.get("c_code", ""),
                    prototype=item.get("prototype", ""),
                    backend=self.name,
                    warnings=item.get("warnings", []),
                )
            else:
                results[va] = DecompiledFunction(
                    name=default_name,
                    va=va,
                    c_code="",
                    prototype="",
                    backend=self.name,
                    warnings=["Function missing from IDA Pro decompilation output"],
                )
        return results
