"""Download the missing public model shards required by the frozen reference."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,default=Path(".hf_cache/hub"))
    parser.add_argument("--repository",default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--endpoint",default="https://hf-mirror.com")
    args=parser.parse_args()
    snapshots=args.cache/("models--"+args.repository.replace("/","--"))/"snapshots"
    snapshot=next(path.parent for path in snapshots.glob("*/model.safetensors.index.json"))
    index=json.loads((snapshot/"model.safetensors.index.json").read_text())
    for name in sorted(set(index["weight_map"].values())):
        target=snapshot/name
        if target.is_file():
            print(f"Already present: {name}",flush=True)
            continue
        partial=target.with_suffix(target.suffix+".partial")
        url=f"{args.endpoint}/{args.repository}/resolve/{snapshot.name}/{name}"
        print(f"Downloading missing shard: {name}",flush=True)
        subprocess.run(["curl","--fail","--location","--retry","2","--connect-timeout","30",
                        "--continue-at","-","--output",str(partial),url],check=True)
        partial.replace(target)
        print(f"Completed {name}: {target.stat().st_size} bytes",flush=True)
    print("Frozen reference checkpoint is available.",flush=True)


if __name__=="__main__":
    main()
