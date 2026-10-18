"""统一滚动修订句子解码与留出测试集评测入口。"""
from __future__ import annotations

import argparse
from importlib import import_module

PROTOCOLS = {
    "validation": "braindecoding.decoding.unified_sentence_v2",
    "heldout_ac": "braindecoding.evaluation.unified_heldout_test_v2",
    "heldout_b": "braindecoding.evaluation.fixed_k0_rolling_search_test_v2_stable",
    "heldout_c1": "braindecoding.evaluation.single_revision_c1_test_v2_stable",
    "heldout_f": "braindecoding.evaluation.full_context_sentence_test_v2",
    "heldout_f_local": "braindecoding.evaluation.f_local_sentence_test_v2",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), default="heldout_ac",
                        help="选择要运行的句子解码协议")
    args, remaining = parser.parse_known_args()
    module = import_module(PROTOCOLS[args.protocol])
    if hasattr(module, 'main'):
        try:
            return int(module.main(remaining) or 0)
        except TypeError:
            return int(module.main() or 0)
    raise SystemExit(f'Module {PROTOCOLS[args.protocol]} has no main()')


if __name__ == "__main__":
    raise SystemExit(main())
