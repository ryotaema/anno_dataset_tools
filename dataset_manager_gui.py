#!/usr/bin/env python3
"""データセット整理ツール（GUI）

    python3 dataset_manager_gui.py

データセットを読み込んで、内訳の確認・まとめ・分割・抽出・検証・変換を
1つの画面から行う。tkinter が必要: sudo apt install python3-tk
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import tkinter  # noqa: F401
except ImportError:
    print('tkinter が必要です: sudo apt install python3-tk')
    sys.exit(1)

from dsm.gui.app import main

if __name__ == '__main__':
    main()
