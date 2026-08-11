#!/usr/bin/env python3
"""動画・bag から学習用の画像を取り出す。

対応する入力:

    動画            mp4 / mov / avi ...
    ROS 2 bag       metadata.yaml のあるフォルダ（画像トピックごとに選べる）
    RealSense bag   収集スクリプトが作る .bag（Color/Depth/IR ごとに選べる）

出力名は収集スクリプトと同じ命名規則にそろえる:

    {cam}_{YYMMDD}_{HHMMSS}_{NNNNN}_{mod}.jpg
    例) nyx_260810_143000_00001_cv.jpg

モダリティは動画・bag 由来を示すため末尾に 'v' が付く
（color → cv / depth → dv / BBox描画済み → detv）。
1つのソース＝1セッションなので、抽出後そのまま dataset_manager_gui.py で
セッション単位の分割（隣接フレームによる情報漏れの防止）ができる。

bag で複数のトピックを選ぶと、基準トピックの時刻に最も近いフレームを他から選び、
同じ連番を振る（生画像と BBox 描画済みが対になる）。

既定は dry-run。実際に書き出すには --apply を付けること。
GUI 版（絵を見ながらトピックを選べます）:
    python3 dataset_manager_gui.py ＞「動画/bag→画像」タブ

使い方:
    # 中身を確認（トピック・ストリームの一覧）
    python3 scripts/extract_frames.py my_rec/ --list

    # 何枚出るか確認（dry-run）
    python3 scripts/extract_frames.py video.mp4 -o frames/

    # 1秒ごと・最大300枚・カメラは nyx
    python3 scripts/extract_frames.py video.mp4 -o frames/ --every 1s --max 300 --cam nyx --apply

    # bag から生画像と BBox 描画済みを取り出し、描画済みは _det/ に分ける
    python3 scripts/extract_frames.py my_rec/ -o frames/ --every 1s \\
        --topics /camera/color/image_raw,/yolo/annotated_image \\
        --mod /yolo/annotated_image=det --subdir /yolo/annotated_image=_det --apply

    # 似すぎ・ブレを弾き、採用しなかった分は 5 秒ごとに _rest/ へ残す
    python3 scripts/extract_frames.py video.mp4 -o frames/ --every 0.5s \\
        --skip-similar --skip-blur --rest thinned:5s --apply

必要なもの:
    動画            opencv        pip install opencv-python
    ROS 2 bag       ROS 2 環境を source したシェルから実行する
    RealSense bag   pyrealsense2  pip install pyrealsense2
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsm import video as V


def parse_every(text):
    """'1s' / '0.5s' / '30f' / '2fps' / 'all' を (mode, value) に変換する。"""
    text = str(text).strip().lower()
    if text in ('all', 'full'):
        return 'all', 0
    m = re.fullmatch(r'([\d.]+)\s*(fps|f|s|sec)?', text)
    if not m:
        raise argparse.ArgumentTypeError(
            f'間隔の書き方が不正です: {text}（例: 1s / 0.5s / 30f / 2fps / all）')
    value, unit = float(m.group(1)), (m.group(2) or 's')
    return {'fps': 'fps', 'f': 'frame', 's': 'sec', 'sec': 'sec'}[unit], value


def parse_rest(text):
    """'discard' / 'all' / 'thinned:5s' を (mode, 秒) に変換する。"""
    text = str(text).strip().lower()
    if text.startswith('thinned'):
        _, _, spec = text.partition(':')
        return 'thinned', (parse_every(spec)[1] if spec else 5.0)
    if text in V.REST_MODES:
        return text, 5.0
    raise argparse.ArgumentTypeError(
        f'残りの扱いが不正です: {text}（discard / all / thinned:5s）')


def build_settings(args):
    mode, value = parse_every(args.every)
    if args.count:
        mode, value = 'count', args.count
    rest_mode, rest_every = parse_rest(args.rest)

    manual_dt, time_source = None, 'auto'
    if args.datetime:
        try:
            manual_dt = datetime.strptime(args.datetime, '%y%m%d_%H%M%S')
        except ValueError:
            try:
                manual_dt = datetime.strptime(args.datetime, '%Y%m%d_%H%M%S')
            except ValueError:
                raise SystemExit(f'--datetime の書き方が不正です: {args.datetime}'
                                 '（YYMMDD_HHMMSS）')
        time_source = 'manual'

    return V.Settings(
        mode=mode, value=value,
        start=V.parse_time(args.start) or 0.0, end=V.parse_time(args.end),
        max_frames=args.max, max_total=args.max_total,
        cam=args.cam, mod=args.mod, video_mod=not args.no_video_suffix,
        time_source=time_source, manual_dt=manual_dt,
        ext=args.ext if args.ext.startswith('.') else '.' + args.ext,
        quality=args.quality, max_side=args.max_side,
        rest=rest_mode, rest_every=rest_every,
        skip_similar=args.skip_similar, similar_thresh=args.similar_thresh,
        skip_blur=args.skip_blur, blur_thresh=args.blur_thresh,
        rest_rejected=not args.drop_rejected,
        align=not args.no_align,
    )


def parse_map(values):
    """['/topic=det', ...] を {'/topic': 'det'} にする。"""
    out = {}
    for item in values or []:
        key, _, value = item.partition('=')
        if not value:
            raise SystemExit(f'書き方が不正です: {item}（トピック名=値 の形で指定してください）')
        out[key.strip()] = value.strip()
    return out


def print_tracks(paths):
    """ソースの中身（トラック一覧）を表示する。"""
    for path in paths:
        kind = V.sources.detect_kind(path)
        tracks, error = V.sources.probe(path)
        print(f'\n{path}  [{V.KIND_LABELS.get(kind, "不明")}]')
        if error:
            print(f'  読み込めません: {error}')
            continue
        if not tracks:
            print('  画像のトラックがありません')
            continue
        print(f'  開始 {tracks[0].started_at:%Y-%m-%d %H:%M:%S}'
              f'（{tracks[0].time_from}） / 長さ {tracks[0].duration_text}')
        print(f'  {"トラック":40} {"枚数":>8} {"fps":>6} {"解像度":>11} {"形式":>8}  推定mod')
        for t in tracks:
            frames = f'{t.n_frames:,}' + ('~' if t.estimated else '')
            print(f'  {(t.key or t.label)[:40]:40} {frames:>8} {t.fps:6.1f} '
                  f'{t.width:>5}x{t.height:<5} {(t.encoding or "-"):>8}  {t.mod_hint}'
                  + ('  ※ 深度なので PNG で保存します' if t.is_depth else ''))
            if t.error:
                print(f'      {t.error}')
        print('  ※ 「推定mod」は名前からの推測です。BBox 描画済みかどうかは名前だけでは'
              '分からないので、\n'
              '     絵を見て選ぶなら GUI の「動画/bag→画像」タブを使ってください。')


def build_choices(paths, args):
    """--topics / --mod / --subdir から抽出対象を決める。"""
    wanted = [t.strip() for t in args.topics.split(',')] if args.topics else []
    mods = parse_map(args.mod_map)
    subdirs = parse_map(args.subdir_map)

    choices, missing = [], list(wanted)
    for path in paths:
        tracks, error = V.sources.probe(path)
        if error or not tracks:
            choices.append(path)              # 計画側で「読めません」として扱う
            continue

        # トラックが複数あるソースは --topics で明示させる。名前からの推測で
        # 勝手に1本選ぶと、BBox 描画済みの方だけを取り出す取り違えが起きる。
        if not wanted and len([t for t in tracks if t.ok]) > 1:
            names = '\n'.join(f'  {t.key or t.label}  （推定 {t.mod_hint}'
                              + (f' / +{t.offset:.1f}s 開始' if t.offset else '')
                              + f' / {t.n_frames:,} 枚）' for t in tracks if t.ok)
            raise SystemExit(
                f'{path} には画像トラックが複数あります。--topics で選んでください:\n'
                f'{names}\n'
                '（名前だけでは生画像か BBox 描画済みか判断できません。'
                '絵を見て選ぶなら GUI の「動画/bag→画像」タブを使ってください）')

        # --topics に書いた順を尊重する（先頭が連番の基準になる）。
        # 指定が無いときは、描画済みらしいトラックを後ろに回して生画像を基準にする。
        if wanted:
            order = sorted((t for t in tracks
                            if t.key in wanted or t.label in wanted),
                           key=lambda t: min(wanted.index(k) for k in (t.key, t.label)
                                             if k in wanted))
        else:
            order = sorted(tracks, key=lambda t: (t.mod_hint == 'det', t.label))

        for t in order:
            if t.key in missing:
                missing.remove(t.key)
            mod = mods.get(t.key) or mods.get(t.label) or (
                args.mod if t.kind == 'video' else t.mod_hint)
            choices.append(V.Choice(track=t, mod=mod,
                                    subdir=subdirs.get(t.key, subdirs.get(t.label, ''))))
    if missing:
        raise SystemExit('指定したトラックが見つかりません: ' + ', '.join(missing)
                         + '\n--list で中身を確認してください。')
    return choices


def main():
    parser = argparse.ArgumentParser(
        description='動画・bag から学習用の画像を取り出す',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='既定は dry-run。実際に書き出すには --apply を付けること。')
    parser.add_argument('path', help='動画 / bag のファイルまたはフォルダ')
    parser.add_argument('-o', '--out', metavar='DIR', help='出力フォルダ（--list 以外で必須）')
    parser.add_argument('--list', action='store_true',
                        help='中身（トピック・ストリーム）を一覧表示して終了する')

    g = parser.add_argument_group('bag のトラック選択')
    g.add_argument('--topics', default='', metavar='A,B',
                   help='抽出するトピック / ストリーム（カンマ区切り）。'
                        '省略時は色画像とみなせるものだけ')
    g.add_argument('--mod', dest='mod_map', action='append', metavar='TOPIC=CODE',
                   help='トラックごとのモダリティ（例: /yolo/annotated_image=det）')
    g.add_argument('--subdir', dest='subdir_map', action='append', metavar='TOPIC=DIR',
                   help='トラックごとの出力サブフォルダ（例: /yolo/annotated_image=_det）')
    g.add_argument('--no-align', action='store_true',
                   help='複数トラックを時刻で対応付けず、それぞれ独立に連番を振る')

    g = parser.add_argument_group('取り出す量')
    g.add_argument('--every', default='1s', metavar='SPEC',
                   help='抽出間隔（1s / 0.5s / 30f / 2fps / all）既定: 1s')
    g.add_argument('--count', type=int, default=0, metavar='N',
                   help='全体から N 枚を等間隔で取る（--every より優先）')
    g.add_argument('--start', default='', metavar='TIME', help='開始位置（秒 または H:MM:SS）')
    g.add_argument('--end', default='', metavar='TIME', help='終了位置（秒 または H:MM:SS）')
    g.add_argument('--max', type=int, default=0, metavar='N', help='1動画あたりの上限枚数')
    g.add_argument('--max-total', type=int, default=0, metavar='N', help='全体の上限枚数')

    g = parser.add_argument_group('名前の付け方')
    g.add_argument('--cam', default='vid', metavar='CODE',
                   help='カメラコード（vid / d435 / d405 / nyx / 任意）既定: vid')
    g.add_argument('--mod-default', dest='mod', default='c', metavar='CODE',
                   help='モダリティ（c=color / d=depth / i1=IR / det=BBox描画済み）既定: c。'
                        'bag は名前から推定するので --mod で個別に上書きする')
    g.add_argument('--no-video-suffix', action='store_true',
                   help="モダリティ末尾に 'v' を付けない（既定は cv のように付ける）")
    g.add_argument('--datetime', default='', metavar='YYMMDD_HHMMSS',
                   help='撮影日時を手で指定する（既定はファイル名→更新時刻の順に判定）')

    g = parser.add_argument_group('出力')
    g.add_argument('--ext', default='.jpg', help='出力形式（.jpg / .png）既定: .jpg')
    g.add_argument('--quality', type=int, default=95, metavar='N', help='JPEG 品質（既定 95）')
    g.add_argument('--max-side', type=int, default=0, metavar='PX',
                   help='長辺をこの画素数まで縮小する（既定: 原寸）')

    g = parser.add_argument_group('採用しなかったフレーム・品質フィルタ')
    g.add_argument('--rest', default='discard', metavar='MODE',
                   help='残りの扱い（discard / all / thinned:5s）既定: discard')
    g.add_argument('--skip-similar', action='store_true', help='似すぎたフレームを飛ばす')
    g.add_argument('--similar-thresh', type=float, default=2.0, metavar='V',
                   help='似すぎと判定する差分の閾値（既定 2.0／大きいほど厳しい）')
    g.add_argument('--skip-blur', action='store_true', help='ブレた画像を弾く')
    g.add_argument('--blur-thresh', type=float, default=100.0, metavar='V',
                   help='ブレと判定する Laplacian 分散の閾値（既定 100）')
    g.add_argument('--drop-rejected', action='store_true',
                   help='フィルタで弾いた分を _rest/ にも残さない')

    parser.add_argument('--apply', action='store_true', help='実際に書き出す（省略時は表示のみ）')
    args = parser.parse_args()

    try:
        V.load_cv2()
    except ImportError as e:
        print(e)
        sys.exit(1)

    found = V.find_sources(args.path)
    if not found:
        print(f'動画・bag が見つかりません: {args.path}\n'
              '対応: 動画（mp4 など） / ROS 2 bag（metadata.yaml のあるフォルダ） / '
              'RealSense の .bag')
        sys.exit(1)

    if args.list:
        print_tracks(found)
        return
    if not args.out:
        print('出力フォルダを -o で指定してください（中身を見るだけなら --list）')
        sys.exit(1)

    settings = build_settings(args)
    choices = build_choices(found, args)
    print(f'ソース {len(found)} 件 / 出力先 {Path(args.out).expanduser()}\n')

    plan = V.build_plan(choices, args.out, settings)
    for path, reason in plan.skipped:
        print(f'[スキップ] {reason}: {path}')
    if plan.skipped:
        print()

    for tp in plan.plans:
        track = tp.track
        print(f'{track.source_label}' + ('   ← 連番の基準' if tp.reference else ''))
        print(f'  {track.duration_text} / {track.fps:.1f}fps / {track.n_frames:,}フレーム '
              f'/ {track.width}x{track.height} / {track.encoding or "-"}')
        print(f'  セッション {tp.session}（日時: {tp.time_from}）')
        print(f'  → {len(tp.picked):,} 枚'
              + (f' ＋ _rest/ に {len(tp.rest):,} 枚' if tp.rest else '')
              + (f'  例) {tp.subdir + "/" if tp.subdir else ""}{tp.first_name}'
                 if tp.picked else ''))
        if tp.note:
            print(f'  [情報] {tp.note}')
        print()

    if plan.conflicts:
        print('[エラー] 出力名が同じになるトラックがあります:')
        for (_, mod, subdir), group in plan.conflicts:
            names = ', '.join(tp.track.key or tp.track.path.name for tp in group)
            print(f'  {mod}' + (f'（{subdir}/）' if subdir else '') + f'  ← {names}')
        print('  --mod か --subdir で分けてください。')
        sys.exit(1)

    if settings.skip_similar or settings.skip_blur:
        print('※ 品質フィルタが有効なため、実際の抽出枚数はこれより少なくなります')
        if V.filter_rest_possible(settings):
            print('   弾いた分は _rest/ に入ります（枚数は実行するまで確定しません）')
        print()

    print(f'合計 {plan.n_picked:,} 枚'
          + (f'（＋ _rest/ {plan.n_rest:,} 枚）' if plan.n_rest else ''))

    if not args.apply:
        print('\n実際に書き出すには --apply を付けて再実行してください。')
        return

    print('\n書き出し中...')
    stats = V.extract(plan, on_log=lambda m: print(m))
    print(f'\n完了: {stats["picked"]:,} 枚を出力'
          + (f' / _rest/ に {stats["rest"]:,} 枚' if stats['rest'] else ''))
    if stats['skipped_blur']:
        print(f'  ブレのため除外: {stats["skipped_blur"]:,} 枚')
    if stats['skipped_similar']:
        print(f'  似すぎたため除外: {stats["skipped_similar"]:,} 枚')
    for path, err in stats['errors'][:10]:
        print(f'  [エラー] {path}: {err}')
    print(f'  設定は {plan.out_dir / "metadata.json"} に保存しました')


if __name__ == '__main__':
    main()
