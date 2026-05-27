# Contributing

Thanks for your interest in the project. The active package lives in
[`manipdyn/`](manipdyn/); the `code/` and `Manipulator Test/` directories are
preserved as earlier prototypes and are not under active development.

## Development setup

All tooling runs from inside the `manipdyn/` directory.

```bash
cd manipdyn
python -m pip install -e ".[gui,rl,dev]"
```

> Run commands from `manipdyn/` rather than the repository root: the root
> `code/` directory shares a name with a Python standard-library module, which
> can shadow it for tools launched at the root.

## Tests, lint, and formatting

```bash
cd manipdyn
pytest                 # full test suite (headless rendering)
ruff check src tests   # lint
ruff format src tests  # format
```

Continuous integration runs the same checks on every push (see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Project conventions

- New controllers implement the `Controller` interface
  (`manipdyn.control.base`); new planners implement `Planner`
  (`manipdyn.planning.base`). Both are picked up by the benchmark and GUI via
  their registries (`CONTROLLERS`, `PLANNERS`).
- Keep documentation in `manipdyn/docs/` in step with code changes.
- Prefer the typed, library-level API over scripts so functionality stays
  testable and reusable.

## Regenerating artifacts

```bash
python scripts/optimize_controllers.py   # tuned controller gains
manipdyn bench                           # benchmark tables + plots
python scripts/make_demos.py             # reach + obstacle demo GIFs
python scripts/make_pick_place.py        # pick-and-place demo GIF
python scripts/train_rl.py               # SAC reaching policy
```
