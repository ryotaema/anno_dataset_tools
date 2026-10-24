#!/usr/bin/env python3
"""ディレクトリ内の画像とラベルの名前を一括で変更する。

セッションが違っても `1color.jpg` のような同じ名前が使われていると、1つの
フォルダに集約した時点で衝突する。ファイル名に「どのデータか」を書き込んでおく。

    260101_lab/color/1color.jpg  →  260101_lab_1color_c.jpg

既定は dry-run。実際に変更するには --apply を付けること。
実行内容は rename_log.json に記録され、--undo でいつでも元に戻せる。

GUI 版: python3 dataset_manager_gui.py ＞「名前変更」タブ

使い方:
    # フォルダから提案される名前を確認（dry-run）
    python3 scripts/rename_images.py 260101_lab/color

    # 先頭と末尾を自分で指定して実行
    python3 scripts/rename_images.py 260101_lab/color \\
        --prefix 260101_lab_ --suffix _c --apply

    # テンプレートで組み立てる（配下のモダリティ別フォルダをまとめて処理）
    python3 scripts/rename_images.py 260101_lab \\
        --template '{parent2}_{stem}_{mod}' --apply

    # 標準命名（{cam}_{YYMMDD}_{HHMMSS}_{NNNNN}_{mod}）に変換する
    python3 scripts/rename_images.py 260101_lab \\
        --standard --cam d435 --datetime 260101_090000 --apply

    # 元に戻す
    python3 scripts/rename_images.py 260101_lab --undo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsm import rename as rn


def show_tokens():
    print('テンプレートで使えるトークン:')
    for token, desc in rn.TOKENS:
        print(f'  {token:<12} {desc}')


def main():
    parser = argparse.ArgumentParser(
        description='画像とラベルの名前を一括で変更する',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='既定は dry-run。実際に変更するには --apply を付けること。')
    parser.add_argument('path', help='対象のフォルダ')

    g = parser.add_argument_group('名前の付け方（指定しなければフォルダ名から提案します）')
    g.add_argument('--prefix', default=None, help='先頭に付ける文字列')
    g.add_argument('--suffix', default=None, help='末尾に付ける文字列')
    g.add_argument('--template', default=None, metavar='TPL',
                   help="テンプレート（例: '{parent2}_{stem}_{mod}'）")
    g.add_argument('--standard', action='store_true',
                   help='標準命名 {cam}_{YYMMDD}_{HHMMSS}_{NNNNN}_{mod} に変換する')
    g.add_argument('--cam', default='', metavar='CODE', help='--standard で使うカメラコード')
    g.add_argument('--datetime', default='', metavar='YYMMDD_HHMMSS',
                   help='--standard で使う日時')
    g.add_argument('--tokens', action='store_true', help='使えるトークンを表示して終了')

    g = parser.add_argument_group('対象')
    g.add_argument('--no-recursive', action='store_true', help='サブフォルダを対象にしない')
    g.add_argument('--no-labels', action='store_true', help='ラベル（.txt/.xml）を変更しない')
    g.add_argument('--labels-dir', default=None, metavar='DIR',
                   help='ラベルの場所を明示する（自動で見つからない構成のとき）')
    g.add_argument('--skip-converted', action='store_true',
                   help='すでに標準命名のファイルは対象にしない')

    g = parser.add_argument_group('実行')
    g.add_argument('--copy-to', default=None, metavar='DIR',
                   help='その場で変えず、新しい名前で別フォルダにコピーする')
    g.add_argument('--no-dedupe', action='store_true',
                   help='名前が重なっても連番で回避しない（衝突として止める）')
    g.add_argument('--lower-ext', action='store_true', help='拡張子を小文字にそろえる')
    g.add_argument('--apply', action='store_true', help='実際に変更する（省略時は表示のみ）')
    g.add_argument('--undo', action='store_true', help='最後の変更を取り消して元に戻す')
    args = parser.parse_args()

    if args.tokens:
        show_tokens()
        return

    root = Path(args.path).expanduser().resolve()
    if not root.is_dir():
        print(f'フォルダが見つかりません: {root}')
        sys.exit(1)

    if args.undo:
        n, msg = rn.undo_last(root)
        print(msg)
        sys.exit(0 if n else 1)

    # 名前の付け方を決める（指定が無ければフォルダ名から提案する）
    hint = rn.suggest(root)
    if args.standard:
        date, _, time = args.datetime.partition('_')
        if not (args.cam and len(date) == 6 and len(time) == 6):
            print('--standard には --cam と --datetime YYMMDD_HHMMSS が必要です')
            sys.exit(1)
        spec = rn.Spec(mode='standard', cam=args.cam, date=date, time=time)
    elif args.template:
        spec = rn.Spec(mode='template', template=args.template)
    else:
        prefix = hint['prefix'] if args.prefix is None else args.prefix
        suffix = hint['suffix'] if args.suffix is None else args.suffix
        if args.prefix is None and args.suffix is None:
            print(f'[提案] {hint["why"]}')
            print(f'       先頭「{prefix}」／末尾「{suffix}」'
                  '  — 変えたいときは --prefix / --suffix で指定してください\n')
        # 先頭・末尾に {mod} などのトークンを書いてもそのまま展開される
        spec = rn.Spec(mode='affix', prefix=prefix, suffix=suffix)

    spec.lower = args.lower_ext
    spec.dedupe = not args.no_dedupe

    plan = rn.build_plan(
        root, spec,
        recursive=not args.no_recursive, include_labels=not args.no_labels,
        labels_dir=args.labels_dir, skip_converted=args.skip_converted,
        mode='copy' if args.copy_to else 'inplace', out_dir=args.copy_to)

    for note in plan.notes:
        print(f'[情報] {note}')

    print(f'対象: {root}')
    print(f'  変更 {plan.n:,} 件（画像 {plan.n_images:,} / ラベル {plan.n_labels:,}）'
          f' / 変更なし {plan.unchanged:,} 件')
    if args.copy_to:
        print(f'  出力先: {Path(args.copy_to).expanduser().resolve()}（原本は変更しません）')
    print()

    for src, dst, kind in plan.pairs[:20]:
        print(f'  [{kind}] {src.name}  →  {dst.name}')
    if plan.n > 20:
        print(f'  ... 他 {plan.n - 20:,} 件')
    print()

    if plan.conflicts:
        print(f'[エラー] 変更後の名前が重複します（{len(plan.conflicts)} 件）:')
        for dst, sources in plan.conflicts[:10]:
            print(f'  {dst}  ←  ' + ', '.join(s.name for s in sources))
        print('  --template で区別できる名前にするか、連番での回避（既定）を使ってください。')
        sys.exit(1)

    if not plan.n:
        print('変更するファイルがありません。')
        return

    if not args.apply:
        print('実際に変更するには --apply を付けて再実行してください。')
        return

    stats = rn.apply_plan(plan)
    print(f'完了: {stats["renamed"]:,} 件を'
          + ('コピーしました' if args.copy_to else '変更しました'))
    for src, err in stats['errors'][:10]:
        print(f'  [エラー] {src}: {err}')
    if stats['log']:
        print(f'  履歴: {stats["log"]}（元に戻すには --undo）')


if __name__ == '__main__':
    main()
