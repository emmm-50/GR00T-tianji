"""Download the official, access-controlled Cosmos backbone into this project."""
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache/huggingface/hub"
REPO = "nvidia/Cosmos-Reason2-2B"
REVISION = "9ce19a195e423419c349abfc86fd07178b230561"


def main():
    snapshot = Path(snapshot_download(
        REPO, revision=REVISION, cache_dir=CACHE, token=True,
        endpoint="https://huggingface.co",
        allow_patterns=["*.json", "*.txt", "*.safetensors"],
        max_workers=3,
    ))
    # The pinned upstream calls from_pretrained(REPO) without a revision.
    ref = snapshot.parent.parent / "refs/main"
    ref.parent.mkdir(exist_ok=True)
    ref.write_text(REVISION)
    print(snapshot)


if __name__ == "__main__":
    main()
