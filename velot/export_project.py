import os
import argparse
from pathlib import Path

DEFAULT_IGNORE = {
    '.git', '__pycache__', '.venv', 'venv', '.env',
    'node_modules', '.pytest_cache', '.vscode',
    'data', 'docs', 'documents', 'experiments',
    'mera', 'export_project.py', 'project_export.txt', 'notes.txt', 'velot_export.txt'
}

DEFAULT_IGNORE_EXTENSIONS = {
    '.pyc', '.pyo', '.pyd', '.so', '.o', '.ipynb', '.txt'
}


def should_ignore(path, ignore_names, ignore_exts):
    return (
        any(part in ignore_names for part in path.parts)
        or path.suffix in ignore_exts
    )


def export_project(base_path, output_file, ignore_names, ignore_exts):
    base_path = Path(base_path).resolve()

    with open(output_file, 'w', encoding='utf-8') as f:
        for path in sorted(base_path.rglob('*')):
            if path.is_file() and not should_ignore(path, ignore_names, ignore_exts):
                rel_path = path.relative_to(base_path)

                f.write(f"\n\n{'='*3} file: {rel_path} {'='*3}\n")

                try:
                    with open(path, 'r', encoding='utf-8') as file:
                        f.write(file.read())
                except Exception:
                    f.write("[Binary or unreadable file]\n")

    print(f"✓ Project exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Export project files into a single text file."
    )

    parser.add_argument(
        "-d", "--dir",
        default=".",
        help="Directory to export (default: current directory)"
    )

    parser.add_argument(
        "-o", "--output",
        default="project_export.txt",
        help="Output file name"
    )

    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        help="Additional folder/file names to ignore"
    )

    parser.add_argument(
        "--ignore-ext",
        nargs="*",
        default=[],
        help="Additional file extensions to ignore (e.g. .csv .png)"
    )

    args = parser.parse_args()

    ignore_names = DEFAULT_IGNORE.union(set(args.ignore))
    ignore_exts = DEFAULT_IGNORE_EXTENSIONS.union(set(args.ignore_ext))

    export_project(
        base_path=args.dir,
        output_file=args.output,
        ignore_names=ignore_names,
        ignore_exts=ignore_exts
    )


if __name__ == "__main__":
    main()