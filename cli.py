#!/usr/bin/env python3
"""Unified CLI for the KV-PR audit and selector pipeline.

Run ``python cli.py <command> --help`` for per-command arguments.
"""

from __future__ import annotations

import argparse
import importlib
import sys


COMMANDS = {
    # Manifest construction.
    "prepare-official-ruler":     "experiments.scripts.prepare_official_ruler",
    "ruler-length-sweep-manifest":"experiments.scripts.ruler_length_sweep_manifest",
    # Audit producer + analyzer + merger.
    "longcontext-decode-probe":           "experiments.scripts.longcontext_decode_probe",
    "longcontext-decode-probe-analyze":   "experiments.scripts.longcontext_decode_probe_analyze",
    "longcontext-decode-probe-merge":     "experiments.scripts.longcontext_decode_probe_merge",
    # End-to-end token-level fidelity probe.
    "e-e2e-accuracy":                     "experiments.scripts.e_e2e_accuracy",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kv-pr",
        description=(
            "KV-PR audit and selector pipeline. "
            "See README.md for a quickstart and docs/reproducibility.md "
            "for the per-figure provenance map."
        ),
    )
    parser.add_argument(
        "command", nargs="?", choices=sorted(COMMANDS),
        help="Pipeline stage to run.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in COMMANDS:
        module = importlib.import_module(COMMANDS[argv[0]])
        module.main(argv[1:])
        return

    args, remainder = build_parser().parse_known_args(argv)
    if args.command is None:
        build_parser().print_help()
        return
    module = importlib.import_module(COMMANDS[args.command])
    module.main(remainder)


if __name__ == "__main__":
    main(sys.argv[1:])
