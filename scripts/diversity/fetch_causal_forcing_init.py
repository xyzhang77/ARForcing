"""Download the pinned official chunk-wise Causal Forcing initialization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

REPO = "zhuhz22/Causal-Forcing"
REVISION = "373037a987c3e06eaab3ec7b2fc2f5c9c296b649"
FILENAME = "chunkwise/causal_ode.pt"
EXPECTED_SIZE = 5676282643
EXPECTED_SHA256 = "34a8aa8e078a7782ad9496991f44daac17c6bb13b83a1214292872186f8a941a"


def verified(path):
    if not path.is_file() or path.stat().st_size != EXPECTED_SIZE:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == EXPECTED_SHA256


def download_http(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".pt.download.part")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}?download=true"
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    raise ValueError("Server did not honor resumable byte range")
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned the wrong resume offset")
                last_report = time.monotonic()
                with partial.open("ab" if offset else "wb") as stream:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        offset += len(block)
                        if time.monotonic() - last_report >= 10:
                            print(json.dumps(dict(downloaded_bytes=offset, total_bytes=EXPECTED_SIZE,
                                                  fraction=round(offset / EXPECTED_SIZE, 4))), flush=True)
                            last_report = time.monotonic()
            if offset != EXPECTED_SIZE:
                raise ValueError(f"Unexpected downloaded length: {offset}")
            if not verified(partial):
                raise ValueError("Downloaded file failed official SHA-256 verification")
            partial.replace(destination)
            return
        except (OSError, TimeoutError) as error:
            print(json.dumps(dict(retry=attempt+1, error=type(error).__name__)), flush=True)
            if attempt == 4:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/causal_forcing_init"))
    parser.add_argument("--transport", choices=("hub", "http"), default="hub")
    parser.add_argument("--finalize-partial", action="store_true",
                        help="Verify and publish a fully downloaded partial file without network access")
    args = parser.parse_args()
    destination = args.output_dir / FILENAME
    already_verified = False
    if args.finalize_partial and not destination.is_file():
        partial = destination.with_suffix(".pt.download.part")
        print("Checking completed partial file against official SHA-256", flush=True)
        if not verified(partial):
            raise ValueError(f"Partial file size/SHA-256 mismatch: {partial}")
        partial.replace(destination)
        already_verified = True
    elif not destination.is_file():
        print(json.dumps(dict(repo=REPO, revision=REVISION, filename=FILENAME,
                              expected_bytes=EXPECTED_SIZE)), flush=True)
        if args.transport == "http":
            download_http(destination)
            already_verified = True
        else:
            # Use the public checkpoint without inheriting an authentication token.
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=REPO, filename=FILENAME, revision=REVISION,
                            local_dir=str(args.output_dir), token=False)
    if not already_verified:
        print("Checking official SHA-256", flush=True)
    if not already_verified and not verified(destination):
        raise ValueError(f"Checkpoint size/SHA-256 mismatch: {destination}")
    metadata = dict(repo_id=REPO, revision=REVISION, filename=FILENAME,
                    size=EXPECTED_SIZE, sha256=EXPECTED_SHA256, complete=True,
                    role="Stage 2 chunk-wise causal ODE initialization, before DMD",
                    official_code="https://github.com/thu-ml/Causal-Forcing",
                    source=f"https://huggingface.co/{REPO}/blob/{REVISION}/{FILENAME}",
                    verified_at_unix=time.time())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / f"source.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(args.output_dir / "source.json")
    print(json.dumps(dict(checkpoint=str(destination.resolve()), **metadata)), flush=True)


if __name__ == "__main__":
    main()
