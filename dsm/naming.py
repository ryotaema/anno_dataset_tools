"""ファイル名から取得元の属性（カメラ・日付・セッション）を読み取る。

real_syutoku / nyx660_syutoku の収集スクリプトが付ける命名規則:

    {cam}_{YYMMDD}_{HHMMSS}_{NNNNN}_{mod}.{ext}
    例) d435_260707_101741_00042_c.jpg

この規則に沿ったファイルなら、学習用に1つのフォルダへ集約したあとでも
「どのカメラの・いつの・どのセッションの何枚目か」を復元できる。
規則に合わない外部データセットや旧命名のファイルは matched=False になり、
カメラ・日付での絞り込みや集計の対象外（'不明'）として扱う。
"""

import re
from dataclasses import dataclass
from datetime import datetime

# {cam}_{YYMMDD}_{HHMMSS}_{NNNNN}_{mod}
NAME_RE = re.compile(
    r'^(?P<cam>[A-Za-z][A-Za-z0-9]*)_(?P<date>\d{6})_(?P<time>\d{6})_'
    r'(?P<shot>\d{5})_(?P<mod>[A-Za-z0-9]+)$'
)

# 旧命名でも日付だけは拾えることがある  例) image_20250911_1544
LEGACY_DATE_RE = re.compile(r'(?<!\d)(?P<date>20\d{6})(?!\d)')

UNKNOWN = '不明'


@dataclass(frozen=True)
class NameInfo:
    """ファイル名から読み取れた属性。読み取れない項目は None。"""
    matched: bool = False
    cam:     str = None        # 'd435' / 'd405' / 'nyx'
    date:    str = None        # 'YYMMDD'
    time:    str = None        # 'HHMMSS'
    shot:    int = None
    mod:     str = None        # 'c' / 'd' / 'dc' / 'i1' ...

    @property
    def session(self):
        """セッションID（{cam}_{YYMMDD}_{HHMMSS}）。不明なら None。"""
        if self.cam and self.date and self.time:
            return f'{self.cam}_{self.date}_{self.time}'
        return None

    @property
    def date_display(self):
        """'26-07-07' 形式。不明なら '不明'。"""
        if not self.date:
            return UNKNOWN
        return f'{self.date[0:2]}-{self.date[2:4]}-{self.date[4:6]}'

    def datetime(self):
        """セッション開始日時。復元できなければ None。"""
        if not (self.date and self.time):
            return None
        try:
            return datetime.strptime(self.date + self.time, '%y%m%d%H%M%S')
        except ValueError:
            return None

    def value(self, key):
        """集計・絞り込み用の値を返す（不明は UNKNOWN）。"""
        if key == 'cam':
            return self.cam or UNKNOWN
        if key == 'date':
            return self.date_display
        if key == 'session':
            return self.session or UNKNOWN
        if key == 'mod':
            return self.mod or UNKNOWN
        return UNKNOWN


NO_INFO = NameInfo()


def parse(stem):
    """ファイル名（拡張子なし）から NameInfo を作る。"""
    m = NAME_RE.match(stem)
    if m:
        return NameInfo(
            matched=True,
            cam=m['cam'].lower(),
            date=m['date'],
            time=m['time'],
            shot=int(m['shot']),
            mod=m['mod'].lower(),
        )

    # 新命名ではないが、YYYYMMDD を含むなら日付だけ拾っておく
    m = LEGACY_DATE_RE.search(stem)
    if m:
        return NameInfo(matched=False, date=m['date'][2:])

    return NO_INFO
