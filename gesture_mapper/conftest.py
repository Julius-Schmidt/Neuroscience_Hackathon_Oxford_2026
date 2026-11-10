"""Lets `pytest` run the standalone check scripts in tests/ (each also runs as `python tests/x.py`).

Every tests/test_*.py becomes one pytest item that runs the script in a subprocess and fails
if it exits non-zero. Qt runs offscreen so the GUI suites need no display.
"""
import os
import subprocess
import sys

import pytest


def pytest_collect_file(file_path, parent):
    if file_path.parent.name == "tests" and file_path.name.startswith("test_") and file_path.suffix == ".py":
        return ScriptFile.from_parent(parent, path=file_path)


class ScriptFile(pytest.File):
    def collect(self):
        yield ScriptItem.from_parent(self, name=self.path.stem)


class ScriptItem(pytest.Item):
    def runtest(self):
        env = dict(os.environ, QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"))
        r = subprocess.run([sys.executable, str(self.path)], cwd=self.path.parent.parent,
                           capture_output=True, text=True, env=env, timeout=300)
        self.output = r.stdout + r.stderr
        if r.returncode != 0:
            raise AssertionError(f"exit code {r.returncode}")

    def repr_failure(self, excinfo):
        return f"{self.path.name} failed:\n{getattr(self, 'output', '')[-3000:]}"

    def reportinfo(self):
        return self.path, None, self.path.name
