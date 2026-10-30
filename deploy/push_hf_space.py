"""Assemble the backend Space and push it to Hugging Face.

    python deploy/push_hf_space.py --stage-only /tmp/space     # just build the folder (for a local docker build)
    HF_TOKEN=hf_... python deploy/push_hf_space.py --repo <user>/redact-api --cors https://<ui>.onrender.com

The Space gets: deploy/hf-space/* at its root, plus src/, api/ and config.yaml. Nothing else (no web/,
data/, tests, keys or notebooks). The token is read from HF_TOKEN or the local `huggingface-cli login`
cache, never from the command line or a file in the repo.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE_FILES = ROOT / "deploy" / "hf-space"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pt", "*.tmp")


def stage(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for f in SPACE_FILES.iterdir():
        shutil.copy2(f, dest / f.name)
    for d in ("src", "api"):
        shutil.copytree(ROOT / d, dest / d, ignore=IGNORE, dirs_exist_ok=True)
    shutil.copy2(ROOT / "config.yaml", dest / "config.yaml")
    leaked = [p for p in dest.rglob("*") if p.name in ("key.bin", ".env") or p.suffix == ".key"]
    if leaked:
        raise SystemExit(f"refusing to stage secrets: {leaked}")
    return dest


def push(repo: str, cors: str, allow_original: bool, staged: Path) -> str:
    from huggingface_hub import HfApi

    api = HfApi()  # HF_TOKEN env var or cached login
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_variable(repo, "CORS_ORIGINS", cors)
    api.add_space_variable(repo, "ALLOW_ORIGINAL_PLAYBACK", "1" if allow_original else "0")
    api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(staged),
                      commit_message="Deploy redaction API backend")
    owner, name = repo.split("/")
    return f"https://{owner.lower()}-{name.lower().replace('_', '-').replace('.', '-')}.hf.space"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-only", metavar="DIR", help="only assemble the Space folder at DIR")
    ap.add_argument("--repo", help="<hf-user>/<space-name>")
    ap.add_argument("--cors", help="comma-separated UI origin(s), e.g. https://redact-ui.onrender.com")
    ap.add_argument("--no-original-playback", action="store_true")
    args = ap.parse_args()
    if args.stage_only:
        print(stage(Path(args.stage_only)))
        return
    if not (args.repo and args.cors):
        ap.error("--repo and --cors are required to push")
    with tempfile.TemporaryDirectory() as tmp:
        url = push(args.repo, args.cors, not args.no_original_playback, stage(Path(tmp)))
    print(f"pushed. The Space builds for ~15-25 min, then serves at {url}")


if __name__ == "__main__":
    main()
