from __future__ import annotations

import sys

from weiss_rl.eval import paper_readiness_check_cli as _cli
from weiss_rl.eval import paper_readiness_check_reporting as _reporting
from weiss_rl.eval.paper_readiness_check_runtime import run_paper_readiness_check

_closed_interval = _cli.closed_interval
_default_readiness_json = _cli.default_readiness_json
_format_alarm = _reporting.format_alarm
_format_alarm_detail = _reporting.format_alarm_detail


def main() -> None:
    parser = _cli.build_paper_readiness_check_parser()
    result = run_paper_readiness_check(parser.parse_args())

    print(f"Paper readiness summary JSON: {result.readiness_json}")
    if result.payload["passed"]:
        print("Paper readiness checks passed.")
        return

    print(_reporting.format_failure_message(result.payload), file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
