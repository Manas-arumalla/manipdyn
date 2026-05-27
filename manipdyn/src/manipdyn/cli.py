"""Command-line interface: ``manipdyn bench | demo | version``."""

from __future__ import annotations

import argparse
from pathlib import Path

from manipdyn import __version__


def _cmd_bench(args: argparse.Namespace) -> None:
    from manipdyn.benchmark import benchmark_controllers, benchmark_planners
    from manipdyn.benchmark.report import markdown_table, write_report

    controller_rows: list[dict] = []
    planner_rows: list[dict] = []

    if args.which in ("controllers", "all"):
        print("Benchmarking controllers (tuned gains)...")
        controller_rows = benchmark_controllers(duration=args.duration)
        print(markdown_table(controller_rows))
        print()
    if args.which in ("planners", "all"):
        print("Benchmarking planners...")
        planner_rows = benchmark_planners(n_trials=args.trials)
        print(markdown_table(planner_rows))
        print()

    report = write_report(controller_rows, planner_rows, Path(args.out))
    print(f"Report written to {report}")


def _cmd_demo(args: argparse.Namespace) -> None:
    import runpy

    # scripts/ lives next to src/, not inside the importable package.
    script = Path(__file__).resolve().parents[2] / "scripts" / "demo_headless.py"
    runpy.run_path(str(script), run_name="__main__")


def _cmd_gui(args: argparse.Namespace) -> None:
    try:
        from manipdyn.gui import launch
    except ImportError as exc:
        raise SystemExit("GUI requires the 'gui' extra: pip install -e '.[gui]'") from exc
    launch()


def _cmd_version(args: argparse.Namespace) -> None:
    print(f"manipdyn {__version__}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="manipdyn", description="6-DOF manipulator control/planning lab"
    )
    sub = parser.add_subparsers(dest="cmd")

    b = sub.add_parser("bench", help="run the benchmark suite")
    b.add_argument("--which", choices=["controllers", "planners", "all"], default="all")
    b.add_argument("--duration", type=float, default=3.0, help="controller rollout seconds")
    b.add_argument("--trials", type=int, default=5, help="planner trials per planner")
    b.add_argument("--out", default="benchmarks/results", help="output directory")
    b.set_defaults(func=_cmd_bench)

    d = sub.add_parser("demo", help="run the headless PID demo (records a GIF)")
    d.set_defaults(func=_cmd_demo)

    g = sub.add_parser("gui", help="launch the PySide6 control center")
    g.set_defaults(func=_cmd_gui)

    v = sub.add_parser("version", help="print version")
    v.set_defaults(func=_cmd_version)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
