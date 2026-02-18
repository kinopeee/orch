#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fake CLI agent for orch integration tests")
    parser.add_argument("subcommand", choices=["inspect", "build", "test"])
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--fail-rate", type=float, default=0.0)
    parser.add_argument("--fail-always", action="store_true")
    parser.add_argument("--produce", type=Path)
    parser.add_argument("--spam-bytes", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sleep > 0:
        time.sleep(args.sleep)

    if args.spam_bytes > 0:
        chunk = ("x" * 128 + "\n").encode("utf-8")
        remaining = args.spam_bytes
        while remaining > 0:
            sys.stdout.buffer.write(chunk[: min(len(chunk), remaining)])
            sys.stdout.flush()
            remaining -= len(chunk)

    payload = {"subcommand": args.subcommand, "timestamp": time.time()}
    print(json.dumps(payload), flush=True)

    if args.produce:
        args.produce.parent.mkdir(parents=True, exist_ok=True)
        args.produce.write_text(json.dumps(payload), encoding="utf-8")

    if args.fail_always:
        print("forced failure", file=sys.stderr, flush=True)
        return 1
    if args.fail_rate > 0 and random.random() < args.fail_rate:
        print("random failure", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
