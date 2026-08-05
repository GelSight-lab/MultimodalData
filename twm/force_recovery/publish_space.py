"""Push the results site to a Hugging Face Space (static SDK)."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

SITE = Path("/media/yxma/Disk1/twm/force_recovery/site")
SPACE_ID = "yxma/react-force-recovery"


def publish() -> str:
    api = HfApi()
    api.create_repo(SPACE_ID, repo_type="space", space_sdk="static",
                    exist_ok=True)
    api.upload_folder(folder_path=str(SITE), repo_id=SPACE_ID,
                      repo_type="space",
                      commit_message="force recovery: methods, evaluation, debug log")
    return f"https://huggingface.co/spaces/{SPACE_ID}"


if __name__ == "__main__":
    print(publish())
