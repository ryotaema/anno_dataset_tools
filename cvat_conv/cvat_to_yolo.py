import argparse
import xml.etree.ElementTree as ET
import os


def read_class_labels(root):
    """XMLのメタデータからクラス名（スケルトンの点などの子ラベルは除く）を定義順に取り出す．

    メタデータが無い場合は，アノテーション中の box の出現順を使う．
    """
    labels = []
    for label in root.findall('./meta//label'):
        name = label.findtext('name')
        if not label.findtext('parent') and name and name not in labels:
            labels.append(name)
    if not labels:
        for box in root.iter('box'):
            name = box.get('label')
            if name and name not in labels:
                labels.append(name)
    return labels


def convert_cvat_to_yolo(xml_file, output_dir, classes=None, include_empty=False):
    """
    CVAT for images 1.1 の XMLファイルを YOLO形式の txtファイルに変換する

    classes を省略すると XML に定義されたラベルを定義順に使う．
    include_empty=True なら box の無い画像にも空の txt を出す（背景画像として学習に使える）．
    使ったクラス一覧を返す．
    """
    # XMLファイルの読み込み
    tree = ET.parse(xml_file)
    root = tree.getroot()

    if classes is None:
        classes = read_class_labels(root)

    # 出力先ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # 画像ごとに処理
    for image in root.findall('image'):
        img_name = image.get('name')
        img_width = float(image.get('width'))
        img_height = float(image.get('height'))

        # 出力ファイル名の作成 (画像の拡張子を .txt に変更)
        txt_name = os.path.splitext(os.path.basename(img_name))[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_name)

        lines = []

        for box in image.findall('box'):
            label = box.get('label')
            if label not in classes:
                continue

            class_id = classes.index(label)

            # CVATの座標 (左上と右下)
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))

            # YOLO形式の計算 (0〜1に正規化)
            x_center = ((xtl + xbr) / 2.0) / img_width
            y_center = ((ytl + ybr) / 2.0) / img_height
            box_width = (xbr - xtl) / img_width
            box_height = (ybr - ytl) / img_height

            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

        # ボックス情報があればファイルに書き込む
        if lines or include_empty:
            with open(txt_path, 'w') as f:
                f.writelines(lines)

    return classes


def read_classes_file(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# --- 実行設定 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CVAT for images 1.1 の XML を YOLO 形式に変換する')
    parser.add_argument('xml', nargs='?', default='annotations.xml',
                        help='CVATから出力したXMLファイル（既定: annotations.xml）')
    parser.add_argument('-o', '--output', default='labels',
                        help='YOLO形式のテキストファイルを出力するフォルダ（既定: labels）')
    parser.add_argument('--classes', metavar='FILE',
                        help='classes.txt（1行1クラス．行の順番がクラスID）．省略時は XML のラベル定義順')
    parser.add_argument('--include-empty', action='store_true',
                        help='box の無い画像にも空の txt を出す')
    args = parser.parse_args()

    classes = read_classes_file(args.classes) if args.classes else None
    classes = convert_cvat_to_yolo(args.xml, args.output, classes, args.include_empty)
    with open(os.path.join(args.output, 'classes.txt'), 'w') as f:
        f.write('\n'.join(classes) + '\n')
    print(f"YOLO形式への変換が完了しました。classes={classes}")
