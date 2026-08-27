"""Stage exactly what the Hugging Face Space needs, into build/space/.

Deploying by hand means deciding each time which files belong in the Space. That
decision drifts, and the failure mode is bad: a Space that boots but silently
serves a stale index, or one that ships the raw corpus and takes minutes to clone.

This script makes the contents a fixed, checked list. It does not push — pushing
publishes under the author's account and needs their credentials.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(".")
STAGE = Path("build/space")

# Copied verbatim.
FILES = [
    "app.py",
]

# The Space gets its own, smaller requirements: it never crawls, never opens a
# PDF and never runs tests, and its torch version is constrained by the hardware
# the Space was created on. Mapping is source path -> name in the staged directory.
RENAMED = {
    "deploy/hf-space/requirements.txt": "requirements.txt",
}

DIRS = [
    "src",
    "data/processed/index",
]

# The Space card carries the YAML frontmatter Hugging Face reads to configure the
# Space, so it becomes the Space's README — not the project README.
SPACE_README = Path("deploy/hf-space/README.md")

EXCLUDE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} TB"


def tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    missing = []
    total = 0

    for name in FILES:
        src = ROOT / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, STAGE / src.name)
        size = tree_size(src)
        total += size
        print(f"  {name:<28} {human(size):>9}")

    for name in DIRS:
        src = ROOT / name
        if not src.exists():
            missing.append(name)
            continue
        dest = STAGE / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest, ignore=EXCLUDE)
        size = tree_size(src)
        total += size
        print(f"  {name + '/':<28} {human(size):>9}")

    for source, target in RENAMED.items():
        src = ROOT / source
        if not src.exists():
            missing.append(source)
            continue
        shutil.copy2(src, STAGE / target)
        size = tree_size(src)
        total += size
        print(f"  {target + ' (space-only)':<28} {human(size):>9}")

    # Written here rather than left to the operator: the index files exceed the
    # Hub's plain-blob limit, and a staging run that forgets this produces a push
    # rejected only at the far end, after the whole upload.
    (STAGE / ".gitattributes").write_text(
        "*.faiss filter=lfs diff=lfs merge=lfs -text\n"
        "*.pkl filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    print(f"  {'.gitattributes (lfs)':<28} {'—':>9}")

    if SPACE_README.exists():
        shutil.copy2(SPACE_README, STAGE / "README.md")
        print(f"  {'README.md (space card)':<28} {human(tree_size(SPACE_README)):>9}")
    else:
        missing.append(str(SPACE_README))

    print(f"\n  {'TOTAL':<28} {human(total):>9}")

    if missing:
        print("\nMUNGON:")
        for name in missing:
            print(f"  - {name}")
        if any("index" in m for m in missing):
            print("\n  Indeksi nuk ekziston. Ndërtoje me scripts/build_index_gpu.py,")
            print("  pastaj `python -m src.index.build --bm25-only`.")
        return

    # The index is what makes the free CPU tier viable; warn if it is only a subset.
    model_stamp = STAGE / "data/processed/index/model.txt"
    if model_stamp.exists():
        print(f"\n  modeli i indeksit: {model_stamp.read_text(encoding='utf-8').strip()}")
    chunks = STAGE / "data/processed/index/article/chunks.jsonl"
    if chunks.exists():
        n = sum(1 for line in chunks.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"  copëza në indeks:  {n:,}")
        if n < 10000:
            print("  KUJDES: indeksi duket si nënkorpus vlerësimi, jo korpusi i plotë.")

    print(f"""
Gati në {STAGE}/

Për ta publikuar (kërkon kredencialet e tua në Hugging Face):

  cd {STAGE}
  git init -b main
  git lfs install
  git lfs track "*.faiss" "*.pkl"
  git add .gitattributes .
  git commit -m "Asistenti Fiskal"
  git remote add origin https://huggingface.co/spaces/aristeaaa/assistent
  git push --force origin main

Pastaj në Space: Settings -> Variables and secrets -> New secret
  ANTHROPIC_API_KEY = <celesi yt>

Pa celes, Space-i funksionon gjithsesi: shfaq nenet e gjetura, pa pergjigje te gjeneruar.
""")


if __name__ == "__main__":
    main()
