# -*- coding: utf-8 -*-
"""
Entry point: safe to run as "python server.py" even when that is Python 2.
Delegates to _server_main.py under Python 3.8+ (or re-execs python3).
"""
from __future__ import print_function

import os
import subprocess
import sys

MIN_PY = (3, 8)
MAIN = "_server_main.py"


def _main_script_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), MAIN)


def _spawn_python3():
    script = _main_script_path()
    argv = [script] + sys.argv[1:]
    for exe in (
        "python3",
        "python3.13",
        "python3.12",
        "python3.11",
        "python3.10",
        "python3.9",
        "python3.8",
    ):
        try:
            raise SystemExit(subprocess.call([exe] + argv))
        except OSError:
            continue
    sys.stderr.write(
        "Python 3.8+ required. Install python3, then: python3 server.py\n"
    )
    raise SystemExit(1)


if __name__ == "__main__":
    if sys.version_info < MIN_PY:
        _spawn_python3()
    import runpy

    runpy.run_path(_main_script_path(), run_name="__main__")
