"""A/C entry: cached preview by default; authorized full evaluation with --full."""
from __future__ import annotations

import os
import argparse


def main(argv=None):
    from braindecoding.config import ensure_default_environment
    ensure_default_environment()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full",action="store_true")
    args = parser.parse_args([] if argv is None else argv)
    if args.full:
        from braindecoding.unified_sentence_v2_revision_gain_full import main as full_main
        full_main([])
    else:
        from braindecoding.unified_sentence_v2_revision_gain import main as revision_gain_main
        revision_gain_main()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
