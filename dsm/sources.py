"""フレームの供給元（動画 / ROS 2 bag / RealSense bag）を1つのインターフェースで扱う。

    動画          1ファイル = 1トラック
    ROS 2 bag     画像トピックごとに1トラック（metadata.yaml + *.db3 / *.mcap）
    RealSense bag RealSense Viewer / SDK で記録した .bag。ストリーム（Color/Depth/IR）ごとに1トラック

BBox を描画した画像を別トピックで流している場合、トピック名だけでは
「どれが生画像でどれが描画済みか」が判断できない。そのため probe() でトラックを
一覧にし、thumbnail() で実際の絵を見てから選べるようにしている。

依存はすべて遅延 import で、使う形式のものだけあれば動く。

    動画            opencv        pip install opencv-python
    ROS 2 bag       rosbag2_py / rclpy（ROS 2 環境を source したシェルから起動する）
    RealSense bag   pyrealsense2  pip install pyrealsense2
"""

import re
import sqlite3
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.mpg', '.mpeg', '.webm', '.wmv'}
RS_BAG_EXTS = {'.bag'}                       # librealsense が書く bag
ROS2_STORAGE_EXTS = {'.db3', '.mcap'}

IMAGE_MSG_TYPES = ('sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage')

KIND_LABELS = {
    'video':     '動画',
    'ros2':      'ROS 2 bag',
    'realsense': 'RealSense bag',
}

# トピック名・ストリーム名からモダリティを推定する（前にあるものから順に判定）。
# あくまで推測なので、GUI では絵を見て直せるようにしている。
_MOD_HINTS = [
    (r'depth[_/]?color|colormap',            'dc'),
    (r'annotat|detect|bbox|box|draw|marked|overlay|render|viz|visual|debug'
     r'|result|predict|infer|yolo|label|track',  'det'),
    (r'infra.*2|ir.*right|right.*ir|infrared_2', 'i2'),
    (r'infra|ir\b|ir[_/]|infrared',          'i1'),
    (r'depth|disparity',                     'd'),
    (r'color|rgb|image_raw|image_rect|bgr',  'c'),
]

# ファイル名から日時を読む（長い形式から順に試す）
_DT_PATTERNS = [
    (re.compile(r'(?<!\d)(\d{8})[_\-]?(\d{6})(?!\d)'), '%Y%m%d%H%M%S'),
    (re.compile(r'(?<!\d)(\d{4}-\d{2}-\d{2})[_\-](\d{6})(?!\d)'), '%Y-%m-%d%H%M%S'),
    (re.compile(r'(?<!\d)(\d{6})[_\-](\d{6})(?!\d)'), '%y%m%d%H%M%S'),
    (re.compile(r'(?<!\d)(\d{8})(?!\d)'), '%Y%m%d'),
]


# --------------------------------------------------------------- 依存の読み込み

def load_cv2():
    """cv2 を読み込む。無ければ分かりやすいメッセージで ImportError。"""
    try:
        import cv2
    except ImportError:
        raise ImportError('画像の読み書きには opencv が必要です: pip install opencv-python')
    return cv2


def load_ros2():
    """ROS 2 bag を読むためのモジュールをまとめて返す。"""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as e:
        raise ImportError(
            'ROS 2 bag を読むには ROS 2 環境が必要です。\n'
            'source /opt/ros/<distro>/setup.bash をしたシェルから起動してください。\n'
            f'（{e}）')
    return rosbag2_py, deserialize_message, get_message


def load_rs():
    """pyrealsense2 を読み込む。"""
    try:
        import pyrealsense2 as rs
    except ImportError:
        raise ImportError('RealSense の bag を読むには pyrealsense2 が必要です: '
                          'pip install pyrealsense2')
    return rs


# --------------------------------------------------------------- トラック

@dataclass
class Track:
    """画像の並び1本。動画は1本、bag はトピック/ストリームごとに1本。"""
    path:       Path                 # ソース（動画ファイル / bag フォルダ / .bag）
    kind:       str = 'video'        # 'video' / 'ros2' / 'realsense'
    key:        str = ''             # ROS 2: トピック名 / RealSense: 'color' など
    label:      str = ''             # 表示名
    n_frames:   int = 0
    fps:        float = 0.0
    width:      int = 0
    height:     int = 0
    encoding:   str = ''             # 'bgr8' / '16UC1' / 'jpeg' / ''
    msg_type:   str = ''             # ROS 2 のみ
    started_at: datetime = None      # このトラックの開始時刻
    time_from:  str = ''             # 開始時刻の由来（表示用）
    stamps:     list = field(default_factory=list)   # 各フレームの相対秒（分かる形式のみ）
    mod_hint:   str = 'c'
    estimated:  bool = False         # 枚数が見積りか
    ok:         bool = True
    error:      str = ''
    handle:     dict = field(default_factory=dict)   # 読み出しに使う内部情報

    @property
    def duration(self):
        """このトラックが続いた長さ（秒）。"""
        if self.stamps:
            return self.stamps[-1] - self.stamps[0]
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def duration_text(self):
        return format_time(self.duration)

    @property
    def offset(self):
        """ソースの先頭から、このトラックが始まるまでの秒数。

        検出ノードを後から起動した場合など、トピックごとに開始が違うことがある。
        """
        return self.stamps[0] if self.stamps else 0.0

    @property
    def is_depth(self):
        """深度データ（jpg にできないので PNG で保存する）か。"""
        return (self.encoding.lower() in ('16uc1', '32fc1', 'mono16', 'z16')
                or self.mod_hint == 'd')

    @property
    def source_label(self):
        return f'{self.path.name}' + (f'  {self.key}' if self.key else '')

    def time_of(self, index):
        """フレーム番号 → トラック開始からの秒数。"""
        if self.stamps:
            i = min(max(index, 0), len(self.stamps) - 1)
            return self.stamps[i]
        return index / self.fps if self.fps else 0.0

    def nearest(self, seconds):
        """指定時刻に最も近いフレーム番号と、そのずれ（秒）を返す。"""
        if not self.n_frames:
            return None, None
        if self.stamps:
            i = bisect_left(self.stamps, seconds)
            best = min((abs(self.stamps[j] - seconds), j)
                       for j in (max(0, i - 1), min(i, len(self.stamps) - 1)))
            return best[1], best[0]
        idx = int(round(seconds * self.fps)) if self.fps else 0
        idx = min(max(idx, 0), self.n_frames - 1)
        return idx, abs(self.time_of(idx) - seconds)


def format_time(sec):
    """秒を H:MM:SS に整形する。"""
    sec = max(0, int(round(sec or 0)))
    return f'{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}'


def modality_hint(name):
    """トピック名・ストリーム名からモダリティコードを推定する。"""
    low = str(name).lower()
    for pattern, code in _MOD_HINTS:
        if re.search(pattern, low):
            return code
    return 'c'


def detect_datetime(path):
    """ファイル名（フォルダ名）から撮影日時を読む。読めなければ None。"""
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


def mtime_of(path):
    path = Path(path)
    if path.is_dir():
        files = [p for p in path.rglob('*') if p.is_file()]
        if files:
            return datetime.fromtimestamp(min(p.stat().st_mtime for p in files))
    return datetime.fromtimestamp(path.stat().st_mtime)


# --------------------------------------------------------------- ソースを探す

def detect_kind(path):
    """パスから形式を判別する。分からなければ ''。"""
    path = Path(path)
    if path.is_dir():
        if (path / 'metadata.yaml').is_file():
            return 'ros2'
        if any(p.suffix.lower() in ROS2_STORAGE_EXTS for p in path.glob('*')):
            return 'ros2'
        return ''
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTS:
        return 'video'
    if suffix in RS_BAG_EXTS:
        return 'realsense'
    if suffix in ROS2_STORAGE_EXTS:
        return 'ros2'
    return ''


def find_sources(path):
    """動画・bag をまとめて探す。ファイル1つならそれだけ。

    ROS 2 bag はフォルダ単位なので、フォルダを渡したときは
    「そのフォルダ自身が bag か」を先に判定する。
    """
    path = Path(path).expanduser()
    if not path.exists():
        return []
    if path.is_file():
        return [path] if detect_kind(path) else []

    if detect_kind(path) == 'ros2':
        return [path]

    found = []
    for p in sorted(path.rglob('*')):
        if p.is_dir():
            if detect_kind(p) == 'ros2':
                found.append(p)
        elif p.suffix.lower() in VIDEO_EXTS or p.suffix.lower() in RS_BAG_EXTS:
            found.append(p)
    # ROS 2 bag の中の .db3 を二重に拾わないよう、bag 配下のファイルを除く
    bags = [p for p in found if p.is_dir()]
    return [p for p in found
            if p.is_dir() or not any(b in p.parents for b in bags)]


# --------------------------------------------------------------- 調べる

def probe(path):
    """ソース1つを調べてトラック一覧を返す。

    Returns:
        (tracks, error)  error は読めなかったときの説明（読めれば ''）
    """
    path = Path(path).expanduser()
    kind = detect_kind(path)
    try:
        if kind == 'video':
            return _probe_video(path), ''
        if kind == 'ros2':
            return _probe_ros2(path), ''
        if kind == 'realsense':
            return _probe_realsense(path), ''
    except ImportError as e:
        return [], str(e)
    except Exception as e:                       # 壊れたファイルでも落とさない
        return [], f'{type(e).__name__}: {e}'
    return [], '対応していない形式です（動画 / ROS 2 bag / RealSense bag）'


def _started(path, prefer=None):
    """開始時刻を (日時, 由来) で返す。"""
    if prefer:
        return prefer, 'bag の記録時刻'
    dt = detect_datetime(path)
    if dt:
        return dt, 'ファイル名'
    return mtime_of(path), '更新時刻'


# ------------------------------------------------ 動画

def _probe_video(path):
    cv2 = load_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [Track(path=path, kind='video', label=path.name, ok=False,
                      error='開けませんでした（対応していない形式かもしれません）')]
    try:
        fps      = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    error = ''
    if fps <= 0 or fps > 1000:
        fps, error = 30.0, 'FPS を読めないため 30 とみなします'
    if n_frames <= 0:
        error = 'フレーム数を読めません（可変フレームレートの可能性）'
        n_frames = 0

    started, time_from = _started(path)
    track = Track(path=path, kind='video', key='', label=path.name,
                  n_frames=n_frames, fps=fps, width=width, height=height,
                  started_at=started, time_from=time_from,
                  mod_hint='c', ok=width > 0 and height > 0, error=error)
    if not track.ok and not track.error:
        track.error = '解像度を読めませんでした'
    return [track]


# ------------------------------------------------ ROS 2 bag

def _ros2_metadata(path):
    """metadata から (トピック情報, 開始日時, 期間秒) を返す。"""
    rosbag2_py, _, _ = load_ros2()
    info = rosbag2_py.Info().read_metadata(str(path), '')
    started = None
    try:
        ns = info.starting_time.nanoseconds
        started = datetime.fromtimestamp(ns / 1e9) if ns > 0 else None
    except Exception:
        started = None
    try:
        duration = info.duration.nanoseconds / 1e9
    except Exception:
        duration = 0.0
    topics = [(t.topic_metadata.name, t.topic_metadata.type, t.message_count)
              for t in info.topics_with_message_count]
    return topics, started, duration


def _db3_files(path):
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() == '.db3' else []
    return sorted(p for p in path.glob('*.db3'))


def _sqlite_rows(files, topic):
    """sqlite3 の bag から、トピックの (ファイル, 行ID, 時刻ns) を時刻順に返す。

    タイムスタンプだけを読むので画像データ（blob）は読み込まない。
    分割された bag（*_0.db3, *_1.db3 ...）にも対応する。
    """
    rows = []
    for f in files:
        try:
            con = sqlite3.connect(f'file:{f}?mode=ro', uri=True)
        except sqlite3.Error:
            continue
        try:
            ids = [r[0] for r in con.execute('SELECT id FROM topics WHERE name=?', (topic,))]
            for tid in ids:
                rows.extend((f, r[0], r[1]) for r in con.execute(
                    'SELECT id, timestamp FROM messages WHERE topic_id=?', (tid,)))
        except sqlite3.Error:
            pass
        finally:
            con.close()
    rows.sort(key=lambda r: r[2])
    return rows


def _probe_ros2(path):
    topics, started, duration = _ros2_metadata(path)
    files = _db3_files(path)
    started, time_from = _started(path, prefer=started)

    image_topics = [(n, t, c) for n, t, c in sorted(topics)
                    if t in IMAGE_MSG_TYPES and c > 0]
    rows_of = {name: (_sqlite_rows(files, name) if files else [])
               for name, _, _ in image_topics}

    # 時刻は bag 全体で共通の原点から数える。トピックごとに「そのトピックの
    # 先頭」を 0 にしてしまうと、あとから始まったトピック（検出ノードを
    # 後で起動した場合など）と時刻がずれ、対応付けが狂う。
    firsts = [rows[0][2] for rows in rows_of.values() if rows]
    base = min(firsts) if firsts else None

    tracks = []
    for name, msg_type, count in image_topics:
        rows = rows_of[name]
        if rows:
            stamps = [(r[2] - base) / 1e9 for r in rows]
            handle = {'rows': [(str(r[0]), r[1]) for r in rows]}
            count = len(rows)
            estimated = False
        else:
            # mcap など sqlite3 でない保存形式。時刻は均等とみなす
            stamps = ([i * duration / max(1, count - 1) for i in range(count)]
                      if duration else [])
            handle = {}
            estimated = True

        span = (stamps[-1] - stamps[0]) if stamps else duration
        fps = (count - 1) / span if span > 0 and count > 1 else 0.0
        track = Track(path=Path(path), kind='ros2', key=name, label=name,
                      n_frames=count, fps=fps, msg_type=msg_type,
                      started_at=started, time_from=time_from, stamps=stamps,
                      mod_hint=modality_hint(name), estimated=estimated,
                      handle=handle)
        _fill_ros2_size(track)
        tracks.append(track)
    return tracks


def _fill_ros2_size(track):
    """先頭の1メッセージだけ読んで解像度とエンコードを埋める。"""
    try:
        for _, img, meta in _read_ros2(track, [0], with_meta=True):
            track.height, track.width = img.shape[:2]
            track.encoding = meta.get('encoding', '')
            if track.mod_hint == 'c' and track.is_depth:
                track.mod_hint = 'd'
            return
    except Exception as e:
        track.ok, track.error = False, f'先頭フレームを読めません: {e}'


def _ros2_message(deserialize, get_message, msg_type, data):
    return deserialize(data, get_message(msg_type))


def _read_ros2(track, indices, with_meta=False):
    """ROS 2 bag から指定 index のフレームを読む。

    sqlite3 の bag では必要な行だけを取り出すので、大きな bag でも
    全体を読み直さずに済む。それ以外の保存形式では順次読みにする。
    """
    rosbag2_py, deserialize, get_message = load_ros2()
    wanted = sorted(set(i for i in indices if 0 <= i < max(1, track.n_frames)))
    if not wanted:
        return

    rows = track.handle.get('rows')
    if rows:
        # 欲しい行だけ file ごとにまとめて取り出す
        by_file = {}
        for i in wanted:
            f, rid = rows[i]
            by_file.setdefault(f, []).append((rid, i))
        got = {}
        for f, pairs in by_file.items():
            ids = [rid for rid, _ in pairs]
            index_of = dict(pairs)
            con = sqlite3.connect(f'file:{f}?mode=ro', uri=True)
            try:
                for chunk in _chunks(ids, 400):
                    q = ('SELECT id, data FROM messages WHERE id IN ('
                         + ','.join('?' * len(chunk)) + ')')
                    for rid, data in con.execute(q, chunk):
                        got[index_of[rid]] = bytes(data)
            finally:
                con.close()
        for i in wanted:
            data = got.get(i)
            if data is None:
                continue
            msg = _ros2_message(deserialize, get_message, track.msg_type, data)
            img, meta = _msg_to_image(msg, track.msg_type)
            yield (i, img, meta) if with_meta else (i, img)
        return

    # 保存形式が sqlite3 でない場合は順次読み（必要な分だけデシリアライズする）
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(track.path), storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[track.key]))
    pos, need = 0, set(wanted)
    limit = wanted[-1]
    while reader.has_next() and pos <= limit:
        _, data, _ = reader.read_next()
        if pos in need:
            msg = _ros2_message(deserialize, get_message, track.msg_type, data)
            img, meta = _msg_to_image(msg, track.msg_type)
            yield (pos, img, meta) if with_meta else (pos, img)
        pos += 1


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _msg_to_image(msg, msg_type):
    """sensor_msgs の画像メッセージを OpenCV の画像にする。

    cv_bridge は使わず、よく使うエンコードを自分で変換する（依存を増やさないため）。
    知らないエンコードのときだけ cv_bridge に任せる。
    Returns: (画像, {'encoding': ...})
    """
    import numpy as np
    cv2 = load_cv2()

    if 'CompressedImage' in msg_type:
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f'圧縮画像を復元できません（format={msg.format}）')
        return img, {'encoding': str(msg.format).split(';')[0].strip()}

    enc = (msg.encoding or '').lower()
    h, w = msg.height, msg.width
    table = {
        'bgr8':  (np.uint8, 3, None),
        'rgb8':  (np.uint8, 3, cv2.COLOR_RGB2BGR),
        'bgra8': (np.uint8, 4, cv2.COLOR_BGRA2BGR),
        'rgba8': (np.uint8, 4, cv2.COLOR_RGBA2BGR),
        'mono8': (np.uint8, 1, None),
        '8uc1':  (np.uint8, 1, None),
        '8uc3':  (np.uint8, 3, None),
        'mono16': (np.uint16, 1, None),
        '16uc1': (np.uint16, 1, None),
        '32fc1': (np.float32, 1, None),
    }
    if enc in table:
        dtype, ch, convert = table[enc]
        arr = np.frombuffer(bytes(msg.data), dtype=dtype)
        arr = arr.reshape(h, w, ch) if ch > 1 else arr.reshape(h, w)
        if convert is not None:
            arr = cv2.cvtColor(arr, convert)
        if enc == '32fc1':                    # メートル → ミリの16bitにそろえる
            arr = np.nan_to_num(arr * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
            arr = arr.astype(np.uint16)
        return arr.copy(), {'encoding': enc}

    if enc.startswith('bayer'):
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w)
        codes = {'bayer_rggb8': cv2.COLOR_BayerBG2BGR, 'bayer_bggr8': cv2.COLOR_BayerRG2BGR,
                 'bayer_gbrg8': cv2.COLOR_BayerGR2BGR, 'bayer_grbg8': cv2.COLOR_BayerGB2BGR}
        code = codes.get(enc)
        if code is not None:
            return cv2.cvtColor(arr, code), {'encoding': enc}

    try:                                       # 知らないエンコードは cv_bridge に任せる
        from cv_bridge import CvBridge
        img = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')
        return img, {'encoding': enc}
    except Exception as e:
        raise ValueError(f'対応していないエンコードです: {msg.encoding}（{e}）')


# ------------------------------------------------ RealSense bag

def _probe_realsense(path):
    rs = load_rs()
    ctx = rs.context()
    dev = ctx.load_device(str(path))
    dev.set_real_time(False)
    playback = dev.as_playback()
    duration = playback.get_duration().total_seconds()

    started, time_from = _started(path)
    tracks, seen = [], set()
    for sensor in dev.query_sensors():
        for prof in sensor.get_stream_profiles():
            video = prof.as_video_stream_profile()
            if video is None:
                continue                      # 画像でないストリーム（IMU など）
            name = prof.stream_name()
            if name in seen:
                continue
            seen.add(name)
            fps = float(prof.fps() or 0)
            handle = {'stream': int(prof.stream_type()), 'index': prof.stream_index()}
            track = Track(
                path=Path(path), kind='realsense', key=name, label=name,
                fps=fps, width=video.width(), height=video.height(),
                encoding=str(prof.format()).replace('format.', ''),
                started_at=started, time_from=time_from,
                mod_hint=modality_hint(name), handle=handle)

            count = _realsense_count(rs, path, handle, duration)
            if count is None:
                track.n_frames = int(round(duration * fps)) if duration and fps else 0
                track.estimated = True
                track.error = '枚数は再生時間からの見積りです（実際の枚数は抽出時に確定）'
            else:
                track.n_frames = count
            tracks.append(track)
    return tracks


def _realsense_count(rs, path, handle, duration):
    """RealSense bag のフレーム数を数える。

    全部再生すると遅いので、先頭と末尾のフレーム番号の差から求める。
    取れなければ None（呼び出し側で再生時間からの見積りにする）。
    """
    from datetime import timedelta
    pipe = rs.pipeline()
    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, str(path), repeat_playback=False)
    stream = rs.stream(handle['stream'])
    cfg.enable_stream(stream, handle.get('index', 0))
    try:
        profile = pipe.start(cfg)
    except RuntimeError:
        return None
    try:
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        first = last = None
        for _ in range(30):
            ok, frames = pipe.try_wait_for_frames(2000)
            if not ok:
                break
            frame = frames.first_or_default(stream)
            if frame:
                first = frame.get_frame_number()
                break
        if first is None:
            return None

        if duration and duration > 0:
            playback.seek(timedelta(seconds=max(0.0, duration - 0.001)))
        for _ in range(1000):
            ok, frames = pipe.try_wait_for_frames(500)
            if not ok:
                break
            frame = frames.first_or_default(stream)
            if frame:
                last = frame.get_frame_number()
        if last is None or last < first:
            return None
        return int(last - first + 1)
    except RuntimeError:
        return None
    finally:
        try:
            pipe.stop()
        except RuntimeError:
            pass


def _read_realsense(track, indices):
    """RealSense bag から指定 index のフレームを読む（先頭から順に再生する）。"""
    import numpy as np
    rs = load_rs()
    wanted = sorted(set(i for i in indices if i >= 0))
    if not wanted:
        return

    pipe = rs.pipeline()
    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, str(track.path), repeat_playback=False)
    cfg.enable_stream(rs.stream(track.handle['stream']), track.handle.get('index', 0))
    profile = pipe.start(cfg)
    profile.get_device().as_playback().set_real_time(False)

    stream_type = rs.stream(track.handle['stream'])
    pos, i, limit = 0, 0, wanted[-1]
    try:
        while i < len(wanted) and pos <= limit:
            ok, frames = pipe.try_wait_for_frames(2000)
            if not ok:
                break
            frame = frames.first_or_default(stream_type)
            if not frame:
                continue
            if pos == wanted[i]:
                img = np.asanyarray(frame.as_video_frame().get_data()).copy()
                yield pos, img
                i += 1
            pos += 1
    finally:
        pipe.stop()


# ------------------------------------------------ 動画の読み出し

def _read_video(track, indices):
    """動画から指定 index のフレームを読む。

    シーク（CAP_PROP_POS_FRAMES）はコーデックによって位置がずれるので使わない。
    不要なフレームは grab()（デコードなし・高速）で読み飛ばす。
    """
    cv2 = load_cv2()
    wanted = sorted(set(i for i in indices if i >= 0))
    if not wanted:
        return
    cap = cv2.VideoCapture(str(track.path))
    if not cap.isOpened():
        return
    try:
        pos, i, limit = 0, 0, wanted[-1]
        while i < len(wanted) and pos <= limit:
            if pos == wanted[i]:
                ok, img = cap.read()
                if not ok:
                    break
                yield pos, img
                i += 1
            elif not cap.grab():
                break
            pos += 1
    finally:
        cap.release()


# --------------------------------------------------------------- 共通の読み出し

def read(track, indices, on_progress=None):
    """トラックから指定 index のフレームを (index, 画像) で順に返す。"""
    readers = {'video': _read_video, 'ros2': _read_ros2, 'realsense': _read_realsense}
    reader = readers.get(track.kind)
    if reader is None:
        return
    total = len(set(indices))
    for n, (index, img) in enumerate(reader(track, indices), 1):
        if on_progress:
            on_progress(n, total)
        yield index, img


def sample_indices(track, count=3):
    """先頭・中間・終盤のように、全体に散らしたフレーム番号を返す（確認用）。"""
    n = track.n_frames
    if n <= 0:
        return [0]
    if n <= count:
        return list(range(n))
    return [min(int(i * (n - 1) / (count - 1)), n - 1) for i in range(count)]
