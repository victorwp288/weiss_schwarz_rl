from __future__ import annotations

from weiss_rl.eval.readiness import fixture_cli as _cli


def main() -> None:
    parser = _cli.build_paper_readiness_fixture_parser()
    result = _cli.run_paper_readiness_fixture_command(parser.parse_args())
    print(f"Wrote paper-readiness fixture run: {result.run_dir}")


if __name__ == "__main__":
    main()
