#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load one behaviour folder.

Run from the repository root:
    python code/load_example.py
    python code/load_example.py walking
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDERS = ("standing", "lying", "walking", "feeding")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "walking"
    if name not in FOLDERS:
        raise SystemExit("choose one of: " + ", ".join(FOLDERS))
    path = os.path.join(ROOT, name, "windows.npy")
    x = np.load(path)
    print(name, x.shape, x.dtype)


if __name__ == "__main__":
    main()
