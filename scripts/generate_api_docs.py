"""Generate API documentation using pdoc."""
from __future__ import annotations

import argparse
from pathlib import Path

import pdoc


def generate(output: Path, module: str = "Medical_KG_rev") -> None:
    context = pdoc.Context()
    module_obj = pdoc.Module(import_path=module, context=context)
    output.mkdir(parents=True, exist_ok=True)
    for page in pdoc.render.HtmlRenderer(context).render(module_obj):
        path = output / page.url
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page.html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate API docs with pdoc")
    parser.add_argument("--output", default="docs/api", help="Output directory")
    parser.add_argument("--module", default="Medical_KG_rev", help="Root module to document")
    args = parser.parse_args()
    generate(Path(args.output), module=args.module)


if __name__ == "__main__":
    main()
