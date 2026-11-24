"""Standalone IDAPython script compatible with IDA 8.x and 9.x.

Exports decompiled C functions into a specified directory (one <func_name>.c per function
plus a manifest.json file) for consumption by newminhh Level 3 re_patcher --decompile-assist.
"""

from __future__ import annotations

import json
import os
import re
import sys

# Attempt importing IDA APIs
try:
    import ida_auto
    import ida_funcs
    import ida_hexrays
    import ida_lines
    import idautils
    import idc
    IN_IDA = True
except ImportError:
    IN_IDA = False

try:
    import ida_kernwin
    HAS_KERNWIN = True
except ImportError:
    HAS_KERNWIN = False


def sanitize_filename(name: str) -> str:
    """Sanitize function name for use as a filesystem filename."""
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)
    return cleaned if cleaned else "unnamed_func"


def get_export_dir() -> str:
    """Determine the destination directory for exported C files."""
    # 1. Environment variable
    env_dir = os.environ.get("IDA_EXPORT_DIR")
    if env_dir:
        return env_dir

    # 2. Script arguments
    args = []
    if IN_IDA and hasattr(idc, "ARGV"):
        args = idc.ARGV
    elif len(sys.argv) > 1:
        args = sys.argv[1:]

    if args and len(args) >= 1 and args[0]:
        return args[0]

    # 3. Interactive GUI prompt if available
    if HAS_KERNWIN and hasattr(ida_kernwin, "ask_dir"):
        chosen = ida_kernwin.ask_dir(".", "Select export directory for decompiled C files")
        if chosen:
            return chosen

    # 4. Fallback to IDB directory
    if IN_IDA:
        idb_path = idc.get_idb_path()
        if idb_path:
            return os.path.join(os.path.dirname(idb_path), "ida_exported_c")

    return os.path.abspath("ida_exported_c")


def main() -> int:
    if not IN_IDA:
        sys.stderr.write(
            "[ida_export_c] ERROR: This script must be run inside IDA Pro / IDA Free (Python 3).\n"
            "Run within IDA via File -> Script file (Alt+F7) or using idat/ida -S\n"
        )
        return 1

    print("[ida_export_c] Waiting for IDA auto-analysis to finish...")
    ida_auto.auto_wait()

    # Initialize Hex-Rays / Cloud Decompiler
    if not ida_hexrays.init_hexrays_plugin():
        msg = (
            "[ida_export_c] WARNING / ERROR: Hex-Rays decompiler is not active!\n"
            "Please ensure the Hex-Rays decompiler plugin (or IDA Free Cloud Decompiler) is loaded.\n"
            "Fallback: You can generate decompiled C manually via File -> Produce file -> Create C file...\n"
        )
        sys.stderr.write(msg)
        if HAS_KERNWIN and hasattr(ida_kernwin, "warning"):
            ida_kernwin.warning(msg)
        return 2

    export_dir = get_export_dir()
    os.makedirs(export_dir, exist_ok=True)
    print(f"[ida_export_c] Exporting decompiled functions to: {export_dir}")

    exported_funcs = []
    failed_funcs = []

    # Iterate over all defined functions
    for func_ea in idautils.Functions():
        func_name = ida_funcs.get_func_name(func_ea) or f"sub_{func_ea:x}"
        safe_name = sanitize_filename(func_name)
        c_filename = f"{safe_name}_{func_ea:x}.c"
        c_filepath = os.path.join(export_dir, c_filename)

        warnings = []
        try:
            cfunc = ida_hexrays.decompile(func_ea)
        except Exception as exc:
            cfunc = None
            warnings.append(f"Decompilation exception: {exc}")

        if cfunc:
            lines = []
            sv = cfunc.get_pseudocode()
            for i in range(len(sv)):
                lines.append(ida_lines.tag_remove(sv[i].line))
            c_code = "\n".join(lines)
            proto = str(cfunc.type) if hasattr(cfunc, "type") else ""

            # Prepend header comment with function metadata
            header = (
                f"/*\n"
                f" * Function: {func_name}\n"
                f" * Virtual Address: 0x{func_ea:x}\n"
                f" * Exported by newminhh ida_export_c.py\n"
                f" */\n\n"
            )

            with open(c_filepath, "w", encoding="utf-8") as f:
                f.write(header + c_code)

            exported_funcs.append({
                "name": func_name,
                "va": func_ea,
                "file": c_filename,
                "prototype": proto,
                "warnings": warnings,
            })
        else:
            failed_funcs.append({
                "name": func_name,
                "va": func_ea,
                "reason": "Hex-Rays decompile returned None",
            })

    # Write manifest.json
    manifest = {
        "generator": "ida_export_c.py",
        "backend": "idafree",
        "total_functions": len(exported_funcs),
        "failed_functions": len(failed_funcs),
        "functions": exported_funcs,
        "failed": failed_funcs,
    }

    manifest_path = os.path.join(export_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"[ida_export_c] Success! Exported {len(exported_funcs)} functions to {export_dir} "
        f"({len(failed_funcs)} failed/skipped). Manifest written to {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    ret = main()
    if not HAS_KERNWIN:
        sys.exit(ret)
