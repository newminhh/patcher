"""Ghidra headless decompiler backend."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from .base import BackendCapability, DecompiledFunction, DecompilerBackend

GHIDRA_SCRIPT_JAVA = """// Ghidra headless decompiler script
// @category NewMinhh
import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.decompiler.DecompiledFunction;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.address.Address;
import java.io.File;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.ArrayList;
import java.util.List;

public class DecompileFunctions extends GhidraScript {
    private String escapeJson(String s) {
        if (s == null) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': sb.append("\\\\\\""); break;
                case '\\\\': sb.append("\\\\\\\\"); break;
                case '\\b': sb.append("\\\\b"); break;
                case '\\f': sb.append("\\\\f"); break;
                case '\\n': sb.append("\\\\n"); break;
                case '\\r': sb.append("\\\\r"); break;
                case '\\t': sb.append("\\\\t"); break;
                default:
                    if (c < 32 || c > 126) {
                        sb.append(String.format("\\\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        return sb.toString();
    }

    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 2) {
            println("Usage: DecompileFunctions <vas_json> <out_json>");
            return;
        }
        String inPath = args[0];
        String outPath = args[1];

        String inputContent = new String(Files.readAllBytes(Paths.get(inPath)));
        List<Long> vas = new ArrayList<>();
        Pattern pat = Pattern.compile("0x[0-9a-fA-F]+|\\\\b\\\\d+\\\\b");
        Matcher matcher = pat.matcher(inputContent);
        while (matcher.find()) {
            String token = matcher.group();
            try {
                if (token.startsWith("0x") || token.startsWith("0X")) {
                    vas.add(Long.parseLong(token.substring(2), 16));
                } else {
                    vas.add(Long.parseLong(token));
                }
            } catch (Exception ex) {
                // ignore
            }
        }

        DecompInterface decomp = new DecompInterface();
        decomp.openProgram(currentProgram);

        FunctionManager fm = currentProgram.getFunctionManager();
        StringBuilder sb = new StringBuilder();
        sb.append("{\\n");
        boolean first = true;

        for (Long va : vas) {
            Address addr = currentProgram.getAddressFactory().getDefaultAddressSpace().getAddress(va);
            Function func = fm.getFunctionAt(addr);
            if (func == null) {
                func = fm.getFunctionContaining(addr);
            }
            if (func == null) {
                try {
                    func = createFunction(addr, null);
                } catch (Exception e) {
                    // ignore
                }
            }

            String funcName = (func != null) ? func.getName() : ("sub_" + Long.toHexString(va));
            String cCode = "";
            String proto = "";
            List<String> warnings = new ArrayList<>();

            if (func != null) {
                DecompileResults res = decomp.decompileFunction(func, 60, monitor);
                if (res != null && res.decompileCompleted()) {
                    DecompiledFunction df = res.getDecompiledFunction();
                    if (df != null) {
                        cCode = df.getC();
                        proto = df.getSignature();
                    }
                    String err = res.getErrorMessage();
                    if (err != null && !err.trim().isEmpty()) {
                        warnings.add(err.trim());
                    }
                } else {
                    warnings.add("Decompilation failed or timed out in Ghidra");
                    if (res != null && res.getErrorMessage() != null) {
                        warnings.add(res.getErrorMessage().trim());
                    }
                }
            } else {
                warnings.add("Function not found at address");
            }

            if (!first) {
                sb.append(",\\n");
            }
            first = false;

            sb.append("  \\"").append(va).append("\\": {\\n");
            sb.append("    \\"name\\": \\"").append(escapeJson(funcName)).append("\\",\\n");
            sb.append("    \\"va\\": ").append(va).append(",\\n");
            sb.append("    \\"c_code\\": \\"").append(escapeJson(cCode)).append("\\",\\n");
            sb.append("    \\"prototype\\": \\"").append(escapeJson(proto)).append("\\",\\n");
            sb.append("    \\"backend\\": \\"ghidra\\",\\n");
            sb.append("    \\"warnings\\": [");
            for (int w = 0; w < warnings.size(); w++) {
                if (w > 0) sb.append(", ");
                sb.append("\\"").append(escapeJson(warnings.get(w))).append("\\"");
            }
            sb.append("]\\n");
            sb.append("  }");
        }

        sb.append("\\n}\\n");
        decomp.dispose();

        try (FileWriter writer = new FileWriter(new File(outPath))) {
            writer.write(sb.toString());
        }
    }
}
"""


class GhidraBackend(DecompilerBackend):
    """Decompiler backend utilizing Ghidra analyzeHeadless."""

    @property
    def name(self) -> str:
        return "ghidra"

    def _find_analyze_headless(self) -> Optional[str]:
        """Locate analyzeHeadless executable from environment or default locations."""
        install_dir = os.environ.get("GHIDRA_INSTALL_DIR")
        if install_dir:
            cand = os.path.join(install_dir, "support", "analyzeHeadless")
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand

        opt_path = "/opt/ghidra/support/analyzeHeadless"
        if os.path.isfile(opt_path) and os.access(opt_path, os.X_OK):
            return opt_path

        which_path = shutil.which("analyzeHeadless")
        if which_path and os.access(which_path, os.X_OK):
            return which_path

        return None

    def probe(self) -> BackendCapability:
        """Probe for Ghidra analyzeHeadless."""
        path = self._find_analyze_headless()
        if path:
            return BackendCapability(
                name=self.name,
                available=True,
                headless=True,
                assist_only=False,
                path=path,
                reason="Ghidra analyzeHeadless detected",
            )
        return BackendCapability(
            name=self.name,
            available=False,
            headless=True,
            assist_only=False,
            path=None,
            reason="Ghidra analyzeHeadless not found (checked GHIDRA_INSTALL_DIR, /opt/ghidra, PATH)",
        )

    def decompile_batch(
        self, elf_path: str, funcs: List[Tuple[int, str]], timeout: int = 120
    ) -> Dict[int, DecompiledFunction]:
        """Decompile functions using headless Ghidra and a temporary Java script."""
        if not funcs:
            return {}

        headless = self._find_analyze_headless()
        if not headless:
            raise RuntimeError("Ghidra analyzeHeadless executable not found")

        with tempfile.TemporaryDirectory(prefix="nm_ghidra_l3_") as tmpdir:
            script_file = os.path.join(tmpdir, "DecompileFunctions.java")
            with open(script_file, "w", encoding="utf-8") as sf:
                sf.write(GHIDRA_SCRIPT_JAVA)

            vas_json = os.path.join(tmpdir, "vas.json")
            with open(vas_json, "w", encoding="utf-8") as vf:
                json.dump([va for va, _ in funcs], vf)

            out_json = os.path.join(tmpdir, "ghidra_out.json")

            cmd = [
                headless,
                tmpdir,
                "nm_proj",
                "-import",
                elf_path,
                "-scriptPath",
                tmpdir,
                "-postScript",
                "DecompileFunctions.java",
                vas_json,
                out_json,
                "-deleteProject",
            ]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            if not os.path.isfile(out_json):
                err = proc.stderr or proc.stdout or ""
                raise RuntimeError(
                    f"Ghidra decompilation failed (rc={proc.returncode}): {err[-500:]}"
                )

            with open(out_json, "r", encoding="utf-8") as jf:
                raw_data = json.load(jf)

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
                        warnings=["Function not found in Ghidra output"],
                    )

            return results
