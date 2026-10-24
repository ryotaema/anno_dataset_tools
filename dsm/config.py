"""個人設定の読み込み．

次の順に探し，最初に見つかった JSON を既定値に上書きする（無ければ既定値のまま）:

    1. 環境変数 DSM_CONFIG で指定したファイル
    2. リポジトリ直下の config.local.json（.gitignore 済み．公開されない）
    3. ~/.config/anno_dataset_tools/config.json

書ける項目は DEFAULTS と config.example.json を参照．
読み込みに失敗しても既定値で動き，理由は WARNINGS に残す（GUI のログに出す）．
"""

import copy
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULTS = {
    # 動画/bag→画像タブ・名前変更タブのカメラ候補（ファイル名からの自動判定にも使う）
    'cameras': ['d435', 'd405', 'nyx'],
    # 各タブのカメラの初期値
    'default_cam': {'video': 'vid', 'rename': 'd435'},
    # フォルダ名・ファイル名 → モダリティコード の追加分（rename.MOD_CODES に足す）
    'mod_codes': {},
    # 変換タブで classes.txt も読み込み済みクラスも無いときに使うクラス一覧
    'classes': [],
    # YOLO-pose 変換のキーポイントの並び順（空なら XML の定義順）
    'keypoints': [],
    # GUI 起動時に自動で読み込むデータセットのフォルダ
    'datasets': [],
}

WARNINGS = []


def candidates():
    env = os.environ.get('DSM_CONFIG')
    if env:
        yield Path(env).expanduser()
    yield REPO / 'config.local.json'
    yield Path.home() / '.config' / 'anno_dataset_tools' / 'config.json'


def load():
    """設定を読み込んで dict を返す．どのファイルを使ったかは 'path' に入る．"""
    cfg = copy.deepcopy(DEFAULTS)
    cfg['path'] = None
    for path in candidates():
        if not path.is_file():
            continue
        try:
            user = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            WARNINGS.append(f'設定ファイルを読めませんでした（既定値で動きます）: {path}: {e}')
            return cfg
        if not isinstance(user, dict):
            WARNINGS.append(f'設定ファイルの中身が {{...}} ではありません: {path}')
            return cfg
        for key, value in user.items():
            if key.startswith('_'):            # "_comment" などは無視
                continue
            if key not in DEFAULTS:
                WARNINGS.append(f'設定ファイルの不明な項目を無視しました: {key}')
            elif not isinstance(value, type(DEFAULTS[key])):
                WARNINGS.append(f'設定ファイルの {key} の型が違うので無視しました'
                                f'（{type(DEFAULTS[key]).__name__} を指定してください）')
            elif not all(isinstance(v, str)
                         for v in (value.values() if isinstance(value, dict) else value)):
                WARNINGS.append(f'設定ファイルの {key} の中身は文字列で書いてください（無視しました）')
            elif isinstance(value, dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
        cfg['path'] = path
        return cfg
    return cfg


CONFIG = load()
