"""動画から学習用の画像を取り出す。

出力名は収集スクリプトと同じ命名規則にそろえる:

    {cam}_{YYMMDD}_{HHMMSS}_{NNNNN}_{mod}.jpg
    例) nyx_260810_143000_00001_cv.jpg

モダリティは動画由来であることを示すため末尾に 'v' を付ける
（color → cv / depth → dv / ir → i1v）。1本の動画＝1セッションとして扱うので、
抽出した直後から概要タブのカメラ・取得日・セッション集計と、
分割タブの「セッション単位」（隣接フレームによる情報漏れの防止）がそのまま効く。

処理は他の操作と同じく「計画を作る（build_plan）」と「書き出す（extract）」に
分けてあり、GUI では枚数を表で確認してから実行できる。

opencv が必要:  pip install opencv-python
"""

import base64
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.mpg', '.mpeg', '.webm', '.wmv'}

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

# モダリティ（動画なので既定は color）。末尾の 'v' は extract 側で付ける
MOD_BASES = {
    'c':  'color（カラー）',
    'd':  'depth（深度）',
    'dc': 'depth_colormap（深度カラーマップ）',
    'i1': 'ir_left（IR左）',
    'i2': 'ir_right（IR右）',
}

CAM_CHOICES = ['vid', 'd435', 'd405', 'nyx']

# ファイル名から日時を読む（長い形式から順に試す）
_DT_PATTERNS = [
    (re.compile(r'(?<!\d)(\d{8})[_\-]?(\d{6})(?!\d)'), '%Y%m%d%H%M%S'),
    (re.compile(r'(?<!\d)(\d{4}-\d{2}-\d{2})[_\-](\d{6})(?!\d)'), '%Y-%m-%d%H%M%S'),
    (re.compile(r'(?<!\d)(\d{6})[_\-](\d{6})(?!\d)'), '%y%m%d%H%M%S'),
    (re.compile(r'(?<!\d)(\d{8})(?!\d)'), '%Y%m%d'),
]


def load_cv2():
    """cv2 を読み込む。無ければ分かりやすいメッセージで ImportError。"""
    try:
        import cv2
    except ImportError:
        raise ImportError('動画の読み込みには opencv が必要です: pip install opencv-python')
    return cv2


# --------------------------------------------------------------- 設定

@dataclass
class Settings:
    """抽出の設定。GUI と CLI で同じものを使う。"""
    # どれだけ取り出すか
    mode:       str   = 'sec'      # INTERVAL_MODES
    value:      float = 1.0
    start:      float = 0.0        # 開始秒
    end:        float = None       # 終了秒（None なら最後まで）
    max_frames: int   = 0          # 1動画あたりの上限（0=無制限）
    max_total:  int   = 0          # 全体の上限（0=無制限）

    # 名前の付け方
    cam:        str = 'vid'
    mod:        str = 'c'
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

    @property
    def mod_code(self):
        return f'{self.mod}v' if self.video_mod else self.mod


# --------------------------------------------------------------- 動画を調べる

@dataclass
class VideoInfo:
    path:     Path
    fps:      float = 0.0
    n_frames: int   = 0
    width:    int   = 0
    height:   int   = 0
    ok:       bool  = False
    error:    str   = ''

    @property
    def duration(self):
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def duration_text(self):
        return format_time(self.duration)


def find_videos(path):
    """動画ファイルを集める。ファイルならそれ1つ、ディレクトリなら再帰で探す。"""
    path = Path(path).expanduser()
    if path.is_file():
        return [path] if path.suffix.lower() in VIDEO_EXTS else []
    if not path.is_dir():
        return []
    return sorted(p for p in path.rglob('*')
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def probe(path):
    """動画の FPS・フレーム数・解像度を読む。開けなければ ok=False。"""
    cv2 = load_cv2()
    path = Path(path)
    info = VideoInfo(path=path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        info.error = '開けませんでした（対応していない形式かもしれません）'
        return info
    try:
        info.fps      = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        info.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        info.width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        info.height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    if info.fps <= 0 or info.fps > 1000:
        info.fps = 30.0
        info.error = 'FPS を読めないため 30 とみなします'
    if info.n_frames <= 0:
        info.error = 'フレーム数を読めません（可変フレームレートの可能性）'
        info.n_frames = 0
    info.ok = info.width > 0 and info.height > 0
    if not info.ok and not info.error:
        info.error = '解像度を読めませんでした'
    return info


def detect_datetime(path):
    """ファイル名から撮影日時を読む。読めなければ None。"""
    stem = Path(path).stem
    for pattern, fmt in _DT_PATTERNS:
        m = pattern.search(stem)
        if not m:
            continue
        try:
            return datetime.strptime(''.join(m.groups()), fmt)
        except ValueError:
            continue
    return None


def detect_cam(path):
    """ファイル名にカメラ名が含まれていればそれを返す。無ければ None。"""
    stem = Path(path).stem.lower()
    for cam in CAM_CHOICES:
        if cam == 'vid':
            continue
        if re.search(rf'(?<![a-z0-9]){cam}(?![a-z0-9])', stem):
            return cam
    return None


def resolve_datetime(path, settings):
    """設定にしたがって開始日時を決める。(日時, 由来の説明) を返す。"""
    src = settings.time_source
    if src == 'manual' and settings.manual_dt:
        return settings.manual_dt, '手入力'
    if src in ('auto', 'filename'):
        dt = detect_datetime(path)
        if dt:
            return dt, 'ファイル名'
        if src == 'filename':
            return _mtime(path), 'ファイル名から読めないため更新時刻'
    return _mtime(path), '更新時刻'


def _mtime(path):
    return datetime.fromtimestamp(Path(path).stat().st_mtime)


def format_time(sec):
    """秒を H:MM:SS に整形する。"""
    sec = max(0, int(round(sec)))
    return f'{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}'


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


# --------------------------------------------------------------- 抽出計画

@dataclass
class VideoPlan:
    """動画1本分の抽出計画。"""
    info:       VideoInfo
    started_at: datetime
    cam:        str = 'vid'
    time_from:  str = ''            # 日時の由来（表示用）
    picked:     list = field(default_factory=list)   # 採用するフレーム番号
    rest:       list = field(default_factory=list)   # _rest/ に出すフレーム番号
    numbers:    dict = field(default_factory=dict)   # フレーム番号 → 連番
    note:       str = ''

    @property
    def session(self):
        return f'{self.cam}_{self.started_at:%y%m%d}_{self.started_at:%H%M%S}'

    def name_of(self, frame, mod, ext):
        return f'{self.session}_{self.numbers[frame]:05d}_{mod}{ext}'

    def first_name(self, mod, ext):
        """先頭の出力名（プレビュー表示用）。"""
        return self.name_of(self.picked[0], mod, ext) if self.picked else ''


@dataclass
class ExtractPlan:
    """複数動画をまとめた抽出計画。"""
    out_dir:  Path
    settings: Settings
    plans:    list = field(default_factory=list)
    skipped:  list = field(default_factory=list)     # (path, 理由)

    @property
    def n_picked(self):
        return sum(len(p.picked) for p in self.plans)

    @property
    def n_rest(self):
        return sum(len(p.rest) for p in self.plans)

    @property
    def rest_dir(self):
        return self.out_dir / '_rest'


def _even_indices(seq, n):
    """seq から n 個を等間隔に取り出す。"""
    if n >= len(seq):
        return list(seq)
    if n <= 0:
        return []
    step = len(seq) / n
    return [seq[min(int(i * step), len(seq) - 1)] for i in range(n)]


def pick_frames(info, settings):
    """1本の動画から採用するフレーム番号を決める。

    Returns:
        (picked, rest, note)  rest は採用しなかったフレーム（設定に応じて間引き済み）
    """
    fps   = info.fps or 30.0
    last  = (info.n_frames - 1) if info.n_frames else 0
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

    # 残り（採用しなかったフレーム）は動画全体から拾う。範囲外の時間帯も
    # 予備プールとして残せるようにするため。
    if settings.rest == 'discard' or not info.n_frames:
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


def build_plan(videos, out_dir, settings):
    """複数の動画から抽出計画を作る。"""
    plan = ExtractPlan(out_dir=Path(out_dir).expanduser(), settings=settings)
    used_sessions = set()
    total = 0

    for path in videos:
        info = probe(path)
        if not info.ok:
            plan.skipped.append((Path(path), info.error or '読み込めません'))
            continue

        started_at, time_from = resolve_datetime(path, settings)
        cam = settings.cam or detect_cam(path) or 'vid'

        # 1本＝1セッション。同じ開始秒の動画があれば1秒ずつずらして一意にする
        while f'{cam}_{started_at:%y%m%d}_{started_at:%H%M%S}' in used_sessions:
            started_at += timedelta(seconds=1)
        used_sessions.add(f'{cam}_{started_at:%y%m%d}_{started_at:%H%M%S}')

        picked, rest, note = pick_frames(info, settings)
        if info.error:
            note = f'{info.error}／{note}' if note else info.error

        # 全体の上限
        if settings.max_total:
            room = settings.max_total - total
            if room <= 0:
                plan.skipped.append((Path(path), f'全体の上限 {settings.max_total} 枚に達しました'))
                continue
            if len(picked) > room:
                picked = _even_indices(picked, room)
                note = (note + '／' if note else '') + f'全体の上限により {room} 枚に制限'
        total += len(picked)

        # 連番は採用分と残りを通しで振る（あとで _rest/ を混ぜても名前が衝突しない）
        numbers = {f: i for i, f in enumerate(sorted(set(picked) | set(rest)), start=1)}

        plan.plans.append(VideoPlan(info=info, started_at=started_at, time_from=time_from,
                                    picked=picked, rest=rest, numbers=numbers,
                                    cam=cam, note=note))
    return plan


# --------------------------------------------------------------- 品質フィルタ

def _gray_small(cv2, frame, size=64):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype('float32')


def blur_score(cv2, frame):
    """Laplacian 分散。小さいほどブレ・ピンぼけ。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def diff_score(prev_small, small):
    """縮小グレースケール同士の平均絶対差（0-255）。小さいほど似ている。"""
    if prev_small is None:
        return 255.0
    return float(abs(small - prev_small).mean())


def sample_quality(plan, n=24, on_progress=None):
    """プレビュー用に候補フレームを少しだけ実測する。

    Returns:
        {'blur': [値...], 'diff': [値...], 'suggest_blur': 推奨閾値}
    """
    cv2 = load_cv2()
    blur, diff = [], []

    # 動画ごとに候補を等間隔でサンプリングする（全部デコードすると遅いため）
    per_video = max(2, n // max(1, len(plan.plans)))
    targets = [(vp, _even_indices(vp.picked, per_video)) for vp in plan.plans]
    total = sum(len(f) for _, f in targets)

    done = 0
    for vp, frames in targets:
        prev = None
        for _, img in _read_frames(cv2, vp.info.path, frames):
            blur.append(blur_score(cv2, img))
            small = _gray_small(cv2, img)
            diff.append(diff_score(prev, small))
            prev = small
            done += 1
            if on_progress:
                on_progress(done, total)

    suggest = 0.0
    if blur:
        s = sorted(blur)
        suggest = round(s[len(s) // 2] * 0.5, 1)      # 中央値の半分
    return {'blur': blur, 'diff': diff, 'suggest_blur': suggest}


# --------------------------------------------------------------- 読み出し

def _read_frames(cv2, path, frames):
    """指定フレーム番号だけをデコードして (フレーム番号, 画像) を順に返す。

    シーク（CAP_PROP_POS_FRAMES）はコーデックによって位置がずれるので使わない。
    不要なフレームは grab()（デコードなし・高速）で読み飛ばす。
    """
    if not frames:
        return
    wanted = sorted(frames)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    try:
        pos, i = 0, 0
        limit = wanted[-1]
        while i < len(wanted) and pos <= limit:
            if pos == wanted[i]:
                ok, img = cap.read()
                if not ok:
                    break
                yield pos, img
                i += 1
            else:
                if not cap.grab():
                    break
            pos += 1
    finally:
        cap.release()


def _encode(cv2, img, settings):
    """設定にしたがってリサイズ・エンコードし、bytes を返す。"""
    if settings.max_side:
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > settings.max_side:
            scale = settings.max_side / longest
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
    ext = settings.ext if settings.ext.startswith('.') else '.' + settings.ext
    params = []
    if ext.lower() in ('.jpg', '.jpeg'):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(settings.quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise OSError(f'エンコードに失敗しました（{ext}）')
    return buf.tobytes()


def thumbnails(plan, count=8, width=200):
    """プレビュー用のサムネイルを PNG の base64 文字列で返す。

    Tk 8.6 の PhotoImage(data=...) にそのまま渡せる形にしておく（PIL 不要）。
    Returns: [(見出し, base64文字列), ...]
    """
    cv2 = load_cv2()
    picks = []
    for vp in plan.plans:
        picks.extend((vp, f) for f in vp.picked)
    if not picks:
        return []
    picks = _even_indices(picks, count)

    out = []
    for vp in plan.plans:
        frames = sorted(f for v, f in picks if v is vp)
        for frame, img in _read_frames(cv2, vp.info.path, frames):
            h, w = img.shape[:2]
            small = cv2.resize(img, (width, max(1, int(round(h * width / w)))),
                               interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode('.png', small)
            if ok:
                label = f'{vp.info.path.name}  f{frame}  ({format_time(frame / (vp.info.fps or 30))})'
                out.append((label, base64.b64encode(buf.tobytes()).decode('ascii')))
    return out


# --------------------------------------------------------------- 書き出し

def extract(plan, on_progress=None, on_log=None):
    """抽出計画を実行してフレームを書き出す。

    Returns:
        {'picked': n, 'rest': n, 'skipped_similar': n, 'skipped_blur': n,
         'errors': [(パス, メッセージ)]}
    """
    cv2 = load_cv2()
    s = plan.settings
    mod, ext = s.mod_code, s.ext if s.ext.startswith('.') else '.' + s.ext
    stats = {'picked': 0, 'rest': 0, 'skipped_similar': 0, 'skipped_blur': 0, 'errors': []}

    total = sum(len(vp.picked) + len(vp.rest) for vp in plan.plans)
    done = 0

    plan.out_dir.mkdir(parents=True, exist_ok=True)
    rest_dir = plan.rest_dir        # 実際に出すときまで作らない（空フォルダを残さない）

    for vp in plan.plans:
        picked_set = set(vp.picked)
        prev_small = None
        if on_log:
            on_log(f'  {vp.info.path.name} → {vp.session}（{len(vp.picked):,} 枚）')

        for frame, img in _read_frames(cv2, vp.info.path, sorted(picked_set | set(vp.rest))):
            done += 1
            to_rest = frame not in picked_set
            reason = ''

            if not to_rest:
                if s.skip_blur:
                    if blur_score(cv2, img) < s.blur_thresh:
                        to_rest, reason = True, 'ブレ'
                        stats['skipped_blur'] += 1
                if not to_rest and s.skip_similar:
                    small = _gray_small(cv2, img)
                    if diff_score(prev_small, small) < s.similar_thresh:
                        to_rest, reason = True, '類似'
                        stats['skipped_similar'] += 1
                    else:
                        prev_small = small

            if to_rest and reason and not s.rest_rejected:
                continue                       # 弾いた分を残さない設定
            if to_rest and not reason and s.rest == 'discard':
                continue

            dst_dir = rest_dir if to_rest else plan.out_dir
            dst = dst_dir / f'{vp.session}_{vp.numbers[frame]:05d}_{mod}{ext}'
            try:
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(_encode(cv2, img, s))
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
        'modality':   s.mod_code,
        'settings':   {k: (v.isoformat() if isinstance(v, datetime) else v)
                       for k, v in asdict(s).items()},
        'result':     {k: v for k, v in stats.items() if k != 'errors'},
        'videos': [{
            'source':     str(vp.info.path),
            'session_id': vp.session,
            'started_at': vp.started_at.isoformat(timespec='seconds'),
            'time_from':  vp.time_from,
            'fps':        round(vp.info.fps, 3),
            'frames':     vp.info.n_frames,
            'size':       f'{vp.info.width}x{vp.info.height}',
            'picked':     len(vp.picked),
            'rest':       len(vp.rest),
        } for vp in plan.plans],
    }
    path = plan.out_dir / 'metadata.json'
    try:
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    except OSError:
        pass
    return path
