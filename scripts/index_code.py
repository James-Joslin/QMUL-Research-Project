"""Build an AST index without importing or executing research modules."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def index(root: Path, baseline: bool = False) -> dict:
    names = subprocess.check_output(["git", "ls-files", "*.py"], cwd=root, text=True).splitlines()
    if not baseline:
        names = sorted(
            {
                *names,
                *(
                    p.relative_to(root).as_posix()
                    for folder in ("src", "tests", "scripts")
                    for p in (root / folder).rglob("*.py")
                ),
            }
        )
    modules, configs, hashes = [], {}, defaultdict(list)
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[digest].append(name)
        symbols = []
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(
                    {"name": node.name, "kind": type(node).__name__, "line": node.lineno}
                )
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append("." * node.level + (node.module or ""))
            if isinstance(node, ast.ClassDef) and node.name == "MuZeroConfig":
                values = {}
                for assignment in ast.walk(node):
                    if isinstance(assignment, ast.Assign):
                        for target in assignment.targets:
                            if (
                                isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                            ):
                                try:
                                    values[target.attr] = ast.literal_eval(assignment.value)
                                except (ValueError, TypeError):
                                    values[target.attr] = {
                                        "expression": ast.unparse(assignment.value)
                                    }
                configs[name] = values
        modules.append(
            {
                "path": name,
                "lines": len(source.splitlines()),
                "sha256": digest,
                "symbols": sorted(symbols, key=lambda s: s["line"]),
                "imports": sorted(set(imports)),
                "entrypoint": any(
                    isinstance(n, ast.Compare) and ast.unparse(n) == "__name__ == '__main__'"
                    for n in ast.walk(tree)
                ),
            }
        )
    checkpoints = [
        {
            "path": p.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in root.glob("muzero-*/results/**/*.checkpoint")
    ]
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "modules": modules,
        "duplicates": [names for names in hashes.values() if len(names) > 1],
        "original_configs": configs,
        "checkpoints": checkpoints,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("docs/code-index.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = index(root, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Indexed {len(result['modules'])} modules, {sum(m['lines'] for m in result['modules'])} lines"
    )


if __name__ == "__main__":
    main()
