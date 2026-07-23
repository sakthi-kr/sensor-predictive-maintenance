from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "validate_leakage_aware_results.py"
)

SPEC = spec_from_file_location(
    "validate_leakage_aware_results",
    MODULE_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(
        f"Could not load validation module: {MODULE_PATH}"
    )

MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_committed_leakage_aware_results_are_consistent() -> None:
    checks = MODULE.run_validation()
    failures = [item for item in checks if not item["passed"]]
    assert not failures, (
        "Leakage-aware result validation failed: "
        f"{failures}"
    )
