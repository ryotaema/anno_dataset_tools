#!/usr/bin/env python3
"""動画（mp4 など）から学習用の画像を取り出す。

出力名は収集スクリプトと同じ命名規則にそろえる:

    {cam}_{YYMMDD}_{HHMMSS}_{NNNNN}_{mod}.jpg
    例) nyx_260810_143000_00001_cv.jpg

モダリティは動画由来を示すため末尾に 'v' が付く（color → cv / depth → dv）。
1本の動画＝1セッションなので、抽出後そのまま dataset_manager_gui.py で
セッション単位の分割（隣接フレームによる情報漏れの防止）ができる。

既定は dry-run。実際に書き出すには --apply を付けること。
GUI 版: python3 dataset_manager_gui.py ＞「動画→画像」タブ

使い方:
    # 何枚出るか確認（dry-run）
    python3 scripts/extract_frames.py video.mp4 -o frames/

    # 1秒ごと・最大300枚・カメラは nyx
    python3 scripts/extract_frames.py video.mp4 -o frames/ --every 1s --max 300 --cam nyx --apply

    # フォルダ内の動画をまとめて、全体から100枚を等間隔に
    python3 scripts/extract_frames.py videos/ -o frames/ --count 100 --apply

    # 似すぎ・ブレを弾き、採用しなかった分は 5 秒ごとに _rest/ へ残す
    python3 scripts/extract_frames.py video.mp4 -o frames/ --every 0.5s \\
        --skip-similar --skip-blur --rest thinned:5s --apply

opencv が必要:  pip install opencv-python
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
    )


def main():
    parser = argparse.ArgumentParser(
        description='動画から学習用の画像を取り出す',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='既定は dry-run。実際に書き出すには --apply を付けること。')
    parser.add_argument('path', help='動画ファイル、または動画を含むフォルダ')
    parser.add_argument('-o', '--out', required=True, metavar='DIR', help='出力フォルダ')

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
    g.add_argument('--mod', default='c', metavar='CODE',
                   help='モダリティ（c=color / d=depth / i1=IR）既定: c')
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

    videos = V.find_videos(args.path)
    if not videos:
        print(f'動画が見つかりません: {args.path}')
        sys.exit(1)

    settings = build_settings(args)
    print(f'動画 {len(videos)} 本 / 出力先 {Path(args.out).expanduser()} '
          f'/ モダリティ {settings.mod_code}\n')

    plan = V.build_plan(videos, args.out, settings)
    for path, reason in plan.skipped:
        print(f'[スキップ] {reason}: {path}')
    if plan.skipped:
        print()

    for vp in plan.plans:
        info = vp.info
        print(f'{info.path.name}')
        print(f'  {info.duration_text} / {info.fps:.1f}fps / {info.n_frames:,}フレーム '
              f'/ {info.width}x{info.height}')
        print(f'  セッション {vp.session}（日時: {vp.time_from}）')
        print(f'  → {len(vp.picked):,} 枚'
              + (f' ＋ _rest/ に {len(vp.rest):,} 枚' if vp.rest else '')
              + (f'  例) {vp.first_name(settings.mod_code, settings.ext)}' if vp.picked else ''))
        if vp.note:
            print(f'  [情報] {vp.note}')
        print()

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
