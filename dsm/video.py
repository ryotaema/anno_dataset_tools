"""動画・bag から学習用の画像を取り出す。

供給元（動画 / ROS 2 bag / RealSense bag）の読み方は dsm/sources.py にあり、
ここでは「どのフレームを何という名前で出すか」を決める。

出力名は収集スクリプトと同じ命名規則にそろえる:

    {cam}_{YYMMDD}_{HHMMSS}_{NNNNN}_{mod}.jpg
    例) nyx_260810_143000_00001_cv.jpg

モダリティは動画・bag 由来であることを示すため末尾に 'v' を付ける
（color → cv / depth → dv / BBox描画済み → detv）。1つのソース＝1セッションとして
扱うので、抽出した直後から概要タブのカメラ・取得日・セッション集計と、
分割タブの「セッション単位」（隣接フレームによる情報漏れの防止）がそのまま効く。

bag で複数のトピックを選んだときは、**基準トラックの時刻に最も近いフレーム**を
他のトラックから選び、同じ連番を振る。生画像と BBox 描画済み画像が
`..._00007_cv.jpg` と `..._00007_detv.jpg` のように対になるので見比べやすく、
分割タブの「ショット単位」でも対が崩れない。

処理は他の操作と同じく「計画を作る（build_plan）」と「書き出す（extract）」に
分けてあり、GUI では枚数を表で確認してから実行できる。
"""

import base64
import json
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

from . import sources
from .sources import (VIDEO_EXTS, KIND_LABELS, Track, load_cv2, format_time,  # noqa: F401
                      detect_datetime, find_sources, probe, sample_indices)

# 抽出間隔の指定方法
INTERVAL_MODES = {
    'sec':   'N 秒ごと',
    'frame': 'N フレームごと',
    'fps':   '毎秒 N 枚（目標FPS）',
    'count': '全体から N 枚を等間隔',
    'all':   '全フレーム',
}

# 採用しなかったフレームの扱い
REST_MODES = {
    'discard': '破棄する',
    'all':     'すべて _rest/ に出す（予備プール）',
    'thinned': '間引いて _rest/ に出す',
}

# モダリティ（既定は color）。末尾の 'v' は Settings.video_mod で付ける
MOD_BASES = {
    'c':   'color（カラー）',
    'd':   'depth（深度）',
    'dc':  'depth_colormap（深度カラーマップ）',
    'i1':  'ir_left（IR左）',
    'i2':  'ir_right（IR右）',
    'det': 'detected（BBox描画済み）',
}

CAM_CHOICES = ['vid', 'd435', 'd405', 'nyx']


def parse_time(text):
    """'12' / '1:30' / '00:01:30' / '1.5' を秒に変換する。空なら None。"""
    text = str(text).strip()
    if not text:
        return None
    if ':' not in text:
        return float(text)
    total = 0.0
    for part in text.split(':'):
        total = total * 60 + float(part or 0)
    return total


def detect_cam(path):
    """ファイル名にカメラ名が含まれていればそれを返す。無ければ None。"""
    import re
    stem = Path(path).stem.lower()
    for cam in CAM_CHOICES:
        if cam == 'vid':
            continue
        if re.search(rf'(?<![a-z0-9]){cam}(?![a-z0-9])', stem):
            return cam
    return None


# --------------------------------------------------------------- 設定

@dataclass
class Settings:
    """抽出の設定。GUI と CLI で同じものを使う。"""
    # どれだけ取り出すか
    mode:       str   = 'sec'      # INTERVAL_MODES
    value:      float = 1.0
    start:      float = 0.0        # 開始秒
    end:        float = None       # 終了秒（None なら最後まで）
    max_frames: int   = 0          # 1ソースあたりの上限（0=無制限）
    max_total:  int   = 0          # 全体の上限（0=無制限）

    # 名前の付け方
    cam:        str = 'vid'
    mod:        str = 'c'          # トラックごとの指定が無いときのモダリティ
    video_mod:  bool = True        # モダリティ末尾に 'v' を付ける
    time_source: str = 'auto'      # 'auto' / 'filename' / 'mtime' / 'manual'
    manual_dt:  datetime = None

    # 出力
    ext:        str = '.jpg'
    quality:    int = 95           # jpg のとき
    max_side:   int = 0            # 長辺をこの画素数に縮小（0=原寸）

    # 採用しなかったフレーム
    rest:       str   = 'discard'  # REST_MODES
    rest_every: float = 5.0        # 'thinned' のときの間隔（秒）

    # 品質フィルタ
    skip_similar:   bool  = False
    similar_thresh: float = 2.0    # 直前の採用フレームとの平均絶対差がこれ未満なら不採用
    skip_blur:      bool  = False
    blur_thresh:    float = 100.0  # Laplacian 分散がこれ未満なら不採用
    rest_rejected:  bool  = True   # フィルタで弾いた分も _rest/ に出す

    # 複数トラック（bag のトピック）を選んだとき
    align:      bool = True        # 時刻で対応付けて同じ連番にする

    def mod_code(self, mod=None):
        base = mod or self.mod
        return f'{base}v' if self.video_mod else base


def filter_rest_possible(settings):
    """品質フィルタで弾いた分が _rest/ に出る設定か。

    「残り: 破棄する」でも、フィルタで弾いた分は _rest/ に残せる。
    プレビューの枚数だけを見ると 0 に見えるので、表示側で補足するために使う。
    """
    return bool((settings.skip_similar or settings.skip_blur) and settings.rest_rejected)


def resolve_datetime(track, settings):
    """設定にしたがって開始日時を決める。(日時, 由来の説明) を返す。"""
    src = settings.time_source
    if src == 'manual' and settings.manual_dt:
        return settings.manual_dt, '手入力'
    if src == 'filename':
        dt = sources.detect_datetime(track.path)
        if dt:
            return dt, 'ファイル名'
        return sources.mtime_of(track.path), 'ファイル名から読めないため更新時刻'
    if src == 'mtime':
        return sources.mtime_of(track.path), '更新時刻'
    # 'auto' … probe が決めた順（bag の記録時刻 → ファイル名 → 更新時刻）に従う
    return track.started_at, track.time_from


# --------------------------------------------------------------- 抽出計画

@dataclass
class Choice:
    """抽出するトラックと、その出し方。"""
    track:  Track
    mod:    str = ''         # 空ならトラックの推定値を使う
    subdir: str = ''         # 出力先のサブフォルダ（'' なら出力フォルダ直下）

    @property
    def label(self):
        return self.track.label


@dataclass
class TrackPlan:
    """トラック1本分の抽出計画。"""
    choice:     Choice
    started_at: datetime
    cam:        str = 'vid'
    mod_code:   str = 'cv'
    ext:        str = '.jpg'
    time_from:  str = ''
    reference:  bool = True                          # 連番の基準になるトラックか
    picked:     list = field(default_factory=list)   # 採用するフレーム番号
    rest:       list = field(default_factory=list)   # _rest/ に出すフレーム番号
    numbers:    dict = field(default_factory=dict)   # フレーム番号 → 連番
    note:       str = ''

    @property
    def track(self):
        return self.choice.track

    @property
    def subdir(self):
        return self.choice.subdir

    @property
    def session(self):
        return f'{self.cam}_{self.started_at:%y%m%d}_{self.started_at:%H%M%S}'

    def name_of(self, frame):
        return f'{self.session}_{self.numbers[frame]:05d}_{self.mod_code}{self.ext}'

    @property
    def first_name(self):
        """先頭の出力名（プレビュー表示用）。"""
        return self.name_of(self.picked[0]) if self.picked else ''


@dataclass
class ExtractPlan:
    """複数のソース・トラックをまとめた抽出計画。"""
    out_dir:  Path
    settings: Settings
    plans:    list = field(default_factory=list)
    skipped:  list = field(default_factory=list)     # (path, 理由)
    conflicts: list = field(default_factory=list)    # 出力名がぶつかるトラックの組

    @property
    def n_picked(self):
        return sum(len(p.picked) for p in self.plans)

    @property
    def n_rest(self):
        return sum(len(p.rest) for p in self.plans)

    @property
    def rest_dir(self):
        return self.out_dir / '_rest'

    def dir_of(self, tp, rest=False):
        """1件の出力先フォルダ。"""
        base = self.rest_dir if rest else self.out_dir
        return base / tp.subdir if tp.subdir else base

    @property
    def references(self):
        return [p for p in self.plans if p.reference]

    @property
    def ok(self):
        return bool(self.plans) and not self.conflicts


def _even_indices(seq, n):
    """seq から n 個を等間隔に取り出す。"""
    if n >= len(seq):
        return list(seq)
    if n <= 0:
        return []
    step = len(seq) / n
    return [seq[min(int(i * step), len(seq) - 1)] for i in range(n)]


def pick_frames(track, settings):
    """1本のトラックから採用するフレーム番号を決める。

    Returns:
        (picked, rest, note)  rest は採用しなかったフレーム（設定に応じて間引き済み）
    """
    fps   = track.fps or 30.0
    last  = (track.n_frames - 1) if track.n_frames else 0
    note  = ''

    f0 = max(0, int(round((settings.start or 0.0) * fps)))
    f1 = last if settings.end is None else min(last, int(round(settings.end * fps)))
    if f1 < f0:
        return [], [], '指定した範囲にフレームがありません'

    span = list(range(f0, f1 + 1))
    mode, value = settings.mode, settings.value

    if mode == 'all':
        picked = span
    elif mode == 'count':
        picked = _even_indices(span, int(value))
    else:
        if mode == 'sec':
            step = int(round(value * fps))
        elif mode == 'fps':
            step = int(round(fps / value)) if value else 1
        else:                                    # 'frame'
            step = int(round(value))
        step = max(1, step)
        picked = span[::step]

    if settings.max_frames and len(picked) > settings.max_frames:
        note = f'上限 {settings.max_frames} 枚に合わせて等間隔に間引きました'
        picked = _even_indices(picked, settings.max_frames)

    # 残り（採用しなかったフレーム）はソース全体から拾う。範囲外の時間帯も
    # 予備プールとして残せるようにするため。
    if settings.rest == 'discard' or not track.n_frames:
        rest = []
    else:
        picked_set = set(picked)
        others = [f for f in range(0, last + 1) if f not in picked_set]
        if settings.rest == 'thinned':
            # 「M 秒ごと」の枚数を目安に、採用しなかったフレームから等間隔で取る。
            # 単純に M 秒周期の番号を拾うと採用フレームと重なって0枚になるため。
            every = max(0.01, settings.rest_every or 1.0)
            rest = _even_indices(others, max(1, int(round((last + 1) / fps / every))))
        else:
            rest = others
    return picked, rest, note


def interval_seconds(track, settings):
    """設定から「1枚あたり何秒間隔か」を求める（時刻の対応付けの許容差に使う）。"""
    fps = track.fps or 30.0
    if settings.mode == 'sec':
        return max(settings.value, 1.0 / fps)
    if settings.mode == 'frame':
        return max(settings.value, 1) / fps
    if settings.mode == 'fps':
        return 1.0 / settings.value if settings.value else 1.0 / fps
    if settings.mode == 'count':
        span = track.duration or (track.n_frames / fps)
        return span / max(1, int(settings.value))
    return 1.0 / fps


def _align_frames(ref, other, ref_frames, numbers, tol, used=None):
    """基準トラックの各フレームに時刻が最も近いフレームを other から選ぶ。

    Args:
        used: すでに割り当てたフレーム番号の集合（その場で更新する）。
              採用分と残りで同じフレームを二重に使わないために共有する。
    Returns:
        (frames, numbers, missed)  missed は近いフレームが無くて欠番にした数
    """
    used = used if used is not None else set()
    frames, mapped, missed = [], {}, 0
    for f in ref_frames:
        t = ref.time_of(f)
        idx, _ = other.nearest(t)
        if idx is None:
            missed += 1
            continue
        # 最も近いフレームが既に使われていたら、その隣まで見て許容差内なら使う
        pick = next((c for c in (idx, idx - 1, idx + 1, idx - 2, idx + 2)
                     if 0 <= c < other.n_frames and c not in used
                     and abs(other.time_of(c) - t) <= tol), None)
        if pick is None:
            missed += 1
            continue
        used.add(pick)
        frames.append(pick)
        mapped[pick] = numbers[f]
    return sorted(frames), mapped, missed


def build_plan(choices, out_dir, settings):
    """抽出計画を作る。

    Args:
        choices: Choice のリスト、または Track / パスのリスト（動画だけのとき）。
    """
    plan = ExtractPlan(out_dir=Path(out_dir).expanduser(), settings=settings)
    choices, skipped = _normalize_choices(choices, settings)
    plan.skipped.extend(skipped)

    # ソース単位にまとめる（同じ bag の複数トピックは同じセッション・同じ連番）
    groups = OrderedDict()
    for ch in choices:
        groups.setdefault(str(ch.track.path), []).append(ch)

    used_sessions, total = set(), 0
    for _, group in groups.items():
        ref = group[0]
        started_at, time_from = resolve_datetime(ref.track, settings)
        cam = settings.cam or detect_cam(ref.track.path) or 'vid'

        # 1ソース＝1セッション。開始秒が同じソースがあれば1秒ずつずらして一意にする
        while f'{cam}_{started_at:%y%m%d}_{started_at:%H%M%S}' in used_sessions:
            started_at += timedelta(seconds=1)
        used_sessions.add(f'{cam}_{started_at:%y%m%d}_{started_at:%H%M%S}')

        picked, rest, note = pick_frames(ref.track, settings)
        if ref.track.error:
            note = f'{ref.track.error}／{note}' if note else ref.track.error

        # 全体の上限は基準トラックの枚数（＝抽出する「瞬間」の数）で数える
        if settings.max_total:
            room = settings.max_total - total
            if room <= 0:
                plan.skipped.append((ref.track.path,
                                     f'全体の上限 {settings.max_total} 枚に達しました'))
                continue
            if len(picked) > room:
                picked = _even_indices(picked, room)
                note = (note + '／' if note else '') + f'全体の上限により {room} 枚に制限'
        total += len(picked)

        # 連番は採用分と残りを通しで振る（あとで _rest/ を混ぜても名前が衝突しない）
        numbers = {f: i for i, f in enumerate(sorted(set(picked) | set(rest)), start=1)}
        plan.plans.append(_track_plan(ref, settings, started_at, time_from, cam,
                                      picked, rest, numbers, True, note))

        for ch in group[1:]:
            if settings.align:
                tol = interval_seconds(ref.track, settings) / 2
                used = set()          # 採用分と残りで同じフレームを取り合わないよう共有する
                a_picked, a_numbers, missed = _align_frames(
                    ref.track, ch.track, picked, numbers, tol, used)
                a_rest, rest_numbers, missed_rest = _align_frames(
                    ref.track, ch.track, rest, numbers, tol, used)
                a_numbers.update(rest_numbers)
                note2 = (f'{missed + missed_rest} 枚は近い時刻のフレームが無いため欠番'
                         if missed + missed_rest else '')
            else:
                a_picked, a_rest, note2 = pick_frames(ch.track, settings)
                a_numbers = {f: i for i, f in
                             enumerate(sorted(set(a_picked) | set(a_rest)), start=1)}
            plan.plans.append(_track_plan(ch, settings, started_at, time_from, cam,
                                          a_picked, a_rest, a_numbers, False, note2))

    # 同じセッション・同じモダリティ・同じ出力先のトラックがあると、
    # 出力名まで同じになって黙って上書きされる。実行前に気付けるようにする。
    seen = OrderedDict()
    for tp in plan.plans:
        seen.setdefault((tp.session, tp.mod_code, tp.subdir), []).append(tp)
    plan.conflicts = [(key, group) for key, group in seen.items() if len(group) > 1]
    return plan


def _track_plan(choice, settings, started_at, time_from, cam,
                picked, rest, numbers, reference, note):
    mod = choice.mod or choice.track.mod_hint or settings.mod
    # 深度は 16bit のまま残したいので、指定が jpg でも PNG で出す
    ext = '.png' if choice.track.is_depth else settings.ext
    if not ext.startswith('.'):
        ext = '.' + ext
    if choice.track.is_depth and settings.ext.lower() in ('.jpg', '.jpeg'):
        note = (note + '／' if note else '') + '深度なので PNG（16bit）で保存します'
    return TrackPlan(choice=choice, started_at=started_at, time_from=time_from, cam=cam,
                     mod_code=settings.mod_code(mod), ext=ext, reference=reference,
                     picked=picked, rest=rest, numbers=numbers, note=note)


def _normalize_choices(items, settings):
    """パス / Track / Choice が混ざったリストを Choice のリストにそろえる。"""
    out, skipped = [], []
    for item in items:
        if isinstance(item, Choice):
            out.append(item)
            continue
        if isinstance(item, Track):
            out.append(Choice(track=item, mod=item.mod_hint))
            continue
        tracks, error = sources.probe(item)
        if error:
            skipped.append((Path(item), error))
            continue
        if not tracks:
            skipped.append((Path(item), '画像のトラックが見つかりません'))
            continue
        for t in tracks:
            if not t.ok:
                skipped.append((Path(item), t.error or '読み込めません'))
                continue
            # 動画は1本だけ。bag は選択が無いので推定モダリティで全部
            out.append(Choice(track=t, mod=(settings.mod if t.kind == 'video'
                                            else t.mod_hint)))
    return out, skipped


# --------------------------------------------------------------- 品質フィルタ

def _to_gray8(cv2, frame):
    """深度（16bit）や白黒でも扱えるように 8bit グレースケールにそろえる。"""
    import numpy as np
    img = frame
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def _gray_small(cv2, frame, size=64):
    gray = _to_gray8(cv2, frame)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype('float32')


def blur_score(cv2, frame):
    """Laplacian 分散。小さいほどブレ・ピンぼけ。"""
    return float(cv2.Laplacian(_to_gray8(cv2, frame), cv2.CV_64F).var())


def diff_score(prev_small, small):
    """縮小グレースケール同士の平均絶対差（0-255）。小さいほど似ている。"""
    if prev_small is None:
        return 255.0
    return float(abs(small - prev_small).mean())


def _sample_blocks(frames, n, blocks=4):
    """候補フレームから、連続したかたまりを全体に散らして取り出す。

    ブレ判定は全体に散らして測りたい一方、似すぎ判定は「隣り合う候補同士」の
    差でなければ意味がないため、等間隔の飛び飛びではなく連続したかたまりで測る。
    """
    if not frames:
        return []
    n = max(1, min(n, len(frames)))
    size = min(max(2, n // max(1, blocks)), len(frames))
    blocks = max(1, min(blocks, len(frames) // size))   # かたまり同士が重ならない数に抑える
    step = len(frames) / blocks

    out, used = [], set()
    for i in range(blocks):
        s = min(int(i * step), len(frames) - size)
        block = tuple(frames[s:s + size])
        if block and block not in used:
            used.add(block)
            out.append(list(block))
    return out


@dataclass
class Measure:
    """品質フィルタの試算のために実測したフレームの指標。

    blocks はソース・かたまりごとの [(ブレ具合, 縮小画像), ...]。
    similar は隣り合うフレームの比較なので、かたまりをまたがずに試算する。
    """
    blocks:     list = field(default_factory=list)
    sampled:    int = 0
    candidates: int = 0        # 実際に抽出する候補フレームの総数

    @property
    def blur_values(self):
        return [b for block in self.blocks for b, _ in block]

    @property
    def diff_values(self):
        """かたまりの中で隣り合うフレームどうしの差。"""
        out = []
        for block in self.blocks:
            prev = None
            for _, small in block:
                if prev is not None:
                    out.append(diff_score(prev, small))
                prev = small
        return out

    def scale(self, n):
        """サンプル中の枚数を全体の枚数に換算する。"""
        if not self.sampled:
            return 0
        return int(round(n / self.sampled * self.candidates))


def measure(plan, n=48, on_progress=None):
    """候補フレームを少しだけ実測して、閾値の試算に使うデータを作る。

    フィルタの判定は基準トラックで行うので、実測も基準トラックだけを見る。
    """
    cv2 = load_cv2()
    targets = plan.references or plan.plans
    per_source = max(4, n // max(1, len(targets)))
    jobs = [(tp, _sample_blocks(tp.picked, per_source)) for tp in targets]
    total = sum(len(b) for _, blocks in jobs for b in blocks) or 1

    result = Measure(candidates=sum(len(tp.picked) for tp in targets))
    done = 0
    for tp, blocks in jobs:
        for frames in blocks:
            seq = []
            for _, img in sources.read(tp.track, frames):
                seq.append((blur_score(cv2, img), _gray_small(cv2, img)))
                done += 1
                if on_progress:
                    on_progress(done, total)
            if seq:
                result.blocks.append(seq)
    result.sampled = done
    return result


def simulate(m, blur_thresh=None, similar_thresh=None):
    """その閾値なら何枚残るかを試算する。

    Returns:
        {'kept': n, 'blur': 弾かれた数, 'similar': 弾かれた数,
         'kept_total': 全体での見込み枚数}
    """
    kept = blur_rej = sim_rej = 0
    for block in m.blocks:
        prev = None
        for blur, small in block:
            if blur_thresh is not None and blur < blur_thresh:
                blur_rej += 1
                continue
            if similar_thresh is not None and diff_score(prev, small) < similar_thresh:
                sim_rej += 1
                continue
            prev = small
            kept += 1
    return {'kept': kept, 'blur': blur_rej, 'similar': sim_rej,
            'kept_total': m.scale(kept)}


def suggest_blur(m):
    """ブレ判定の閾値の目安（実測の中央値の半分）。"""
    values = sorted(m.blur_values)
    if not values:
        return 0.0
    return round(values[len(values) // 2] * 0.5, 1)


def stats_text(values):
    """最小 / 中央 / 最大 を1行にまとめる（説明の表示用）。"""
    if not values:
        return '—'
    s = sorted(values)
    return f'最小 {s[0]:.1f} / 中央 {s[len(s) // 2]:.1f} / 最大 {s[-1]:.1f}'


# --------------------------------------------------------------- サムネイル

def to_png_base64(cv2, img, width=200):
    """Tk の PhotoImage(data=...) に渡せる PNG の base64 文字列にする。"""
    import numpy as np
    if img.dtype != np.uint8:                 # 深度は見えるように正規化する
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    h, w = img.shape[:2]
    small = cv2.resize(img, (width, max(1, int(round(h * width / w)))),
                       interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.png', small)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode('ascii')


def thumbnails(plan, count=8, width=200, on_progress=None):
    """プレビュー用のサムネイルを [(見出し, base64), ...] で返す。"""
    cv2 = load_cv2()
    picks = []
    for tp in plan.plans:
        picks.extend((tp, f) for f in tp.picked)
    if not picks:
        return []
    picks = _even_indices(picks, count)

    out = []
    for tp in plan.plans:
        frames = sorted(f for p, f in picks if p is tp)
        for frame, img in sources.read(tp.track, frames):
            if on_progress:
                on_progress(len(out) + 1, len(picks))
            data = to_png_base64(cv2, img, width)
            if data:
                label = (f'{tp.mod_code}  f{frame}  '
                         f'({format_time(tp.track.time_of(frame))})')
                out.append((label, data))
    return out


def track_thumbnails(track, indices=None, width=260):
    """トラック選択画面用。指定フレーム（既定は先頭・中間・終盤）の絵を返す。"""
    cv2 = load_cv2()
    indices = indices if indices is not None else sample_indices(track, 3)
    out = []
    for frame, img in sources.read(track, indices):
        data = to_png_base64(cv2, img, width)
        if data:
            out.append((f'f{frame}  ({format_time(track.time_of(frame))})', data))
    return out


# --------------------------------------------------------------- 書き出し

def _encode(cv2, img, settings, ext):
    """設定にしたがってリサイズ・エンコードし、bytes を返す。"""
    import numpy as np
    if settings.max_side:
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > settings.max_side:
            scale = settings.max_side / longest
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_NEAREST if img.dtype != np.uint8
                             else cv2.INTER_AREA)
    params = []
    if ext.lower() in ('.jpg', '.jpeg'):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(settings.quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise OSError(f'エンコードに失敗しました（{ext}）')
    return buf.tobytes()


def extract(plan, on_progress=None, on_log=None):
    """抽出計画を実行してフレームを書き出す。

    複数トラックを時刻で対応付けている場合、品質フィルタの判定は基準トラックで
    行い、同じ連番の他トラックにも同じ判定を適用する（対が崩れないように）。

    Returns:
        {'picked': n, 'rest': n, 'skipped_similar': n, 'skipped_blur': n,
         'errors': [(パス, メッセージ)]}
    """
    cv2 = load_cv2()
    s = plan.settings
    stats = {'picked': 0, 'rest': 0, 'skipped_similar': 0, 'skipped_blur': 0, 'errors': []}

    total = sum(len(tp.picked) + len(tp.rest) for tp in plan.plans)
    done = 0
    plan.out_dir.mkdir(parents=True, exist_ok=True)

    # 基準トラックを先に処理して、連番ごとの判定（採用 / _rest / 破棄）を作る
    verdicts = {}
    for tp in sorted(plan.plans, key=lambda p: not p.reference):
        session_verdict = verdicts.setdefault(tp.session, {})
        picked_set = set(tp.picked)
        prev_small = None
        if on_log:
            on_log(f'  {tp.track.source_label} → {tp.session} '
                   f'[{tp.mod_code}]（{len(tp.picked):,} 枚）')

        for frame, img in sources.read(tp.track, sorted(picked_set | set(tp.rest))):
            done += 1
            number = tp.numbers.get(frame)
            to_rest = frame not in picked_set
            reason = ''

            if not to_rest:
                if tp.reference or not s.align:
                    if s.skip_blur and blur_score(cv2, img) < s.blur_thresh:
                        to_rest, reason = True, 'ブレ'
                        stats['skipped_blur'] += 1
                    if not to_rest and s.skip_similar:
                        small = _gray_small(cv2, img)
                        if diff_score(prev_small, small) < s.similar_thresh:
                            to_rest, reason = True, '類似'
                            stats['skipped_similar'] += 1
                        else:
                            prev_small = small
                    session_verdict[number] = ('rest' if to_rest else 'out', reason)
                else:
                    # 基準トラックの判定に合わせる（同じ瞬間の画像なので対で扱う）
                    kind, reason = session_verdict.get(number, ('out', ''))
                    to_rest = kind == 'rest'

            if to_rest and reason and not s.rest_rejected:
                continue                       # 弾いた分を残さない設定
            if to_rest and not reason and s.rest == 'discard':
                continue

            dst_dir = plan.dir_of(tp, rest=to_rest)
            dst = dst_dir / tp.name_of(frame)
            try:
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(_encode(cv2, img, s, tp.ext))
                stats['rest' if to_rest else 'picked'] += 1
            except OSError as e:
                stats['errors'].append((dst, str(e)))

            if on_progress and (done % 20 == 0 or done == total):
                on_progress(done, total)

    write_metadata(plan, stats)
    return stats


def write_metadata(plan, stats):
    """出力先に metadata.json を書く（同じ条件で追加抽出できるように）。"""
    s = plan.settings
    meta = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'naming':     '{cam}_{YYMMDD}_{HHMMSS}_{NNNNN}_{mod}.{ext}',
        'settings':   {k: (v.isoformat() if isinstance(v, datetime) else v)
                       for k, v in asdict(s).items()},
        'result':     {k: v for k, v in stats.items() if k != 'errors'},
        'tracks': [{
            'source':     str(tp.track.path),
            'kind':       tp.track.kind,
            'track':      tp.track.key,
            'session_id': tp.session,
            'modality':   tp.mod_code,
            'subdir':     tp.subdir,
            'reference':  tp.reference,
            'started_at': tp.started_at.isoformat(timespec='seconds'),
            'time_from':  tp.time_from,
            'fps':        round(tp.track.fps, 3),
            'frames':     tp.track.n_frames,
            'size':       f'{tp.track.width}x{tp.track.height}',
            'encoding':   tp.track.encoding,
            'picked':     len(tp.picked),
            'rest':       len(tp.rest),
        } for tp in plan.plans],
    }
    path = plan.out_dir / 'metadata.json'
    try:
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    except OSError:
        pass
    return path
