"""
Pytest hooks for test_preprocess_mining_outputs.py.

- Failed data-quality checks show their compact message instead of pytest's traceback
  (no source listing, no "E" prefixes).
- The end of the run lists the data-quality warnings and the published data compared
  against, and points to the Markdown report.

The test module does the work (REPORT, console_summary, write_report); these hooks
only call it. They apply to tests in this folder only.
"""

import os
import shutil

import pytest

# nodeid -> passed / failed / skipped, for the report's checklist
OUTCOMES = {}
# (test module, report path), filled in at the end of the session
_RESULT = {}


def pytest_configure(config):
    # Output isn't captured yet here, so this sees the real terminal. The failure
    # messages are built while output is captured, where the width can't be read, so
    # they get it from COLUMNS.
    os.environ.setdefault("COLUMNS", str(shutil.get_terminal_size((120, 40)).columns))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    # matched by name: importing the test module here would load a second copy of it
    if report.failed and call.excinfo is not None:
        if type(call.excinfo.value).__name__ == "DataQualityError":
            report.longrepr = str(call.excinfo.value)


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and not report.passed):
        OUTCOMES[report.nodeid] = report.outcome


def pytest_sessionfinish(session):
    items = getattr(session, "items", [])
    modules = {item.module for item in items if hasattr(item.module, "write_report")}
    for module in modules:
        module_items = [item for item in items if item.module is module]
        _RESULT["module"] = module
        _RESULT["path"] = module.write_report(module_items, OUTCOMES)


def pytest_terminal_summary(terminalreporter):
    module = _RESULT.get("module")
    if module is None:
        return
    lines = module.console_summary()
    if lines:
        terminalreporter.section("data-quality warnings and comparison")
        for line in lines:
            terminalreporter.write_line(line)
    terminalreporter.write_line(f"Full report: {_RESULT['path']}", bold=True)