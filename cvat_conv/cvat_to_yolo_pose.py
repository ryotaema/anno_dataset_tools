import argparse
import xml.etree.ElementTree as ET
import os


def read_skeletons(root):
    """XMLのメタデータから {スケルトン名: [キーポイント名, ...]} を定義順に取り出す．

    CVATのXMLでは，スケルトンの各点が <parent> 付きの <label> として
    <meta> に書かれている．メタデータが無い場合はアノテーション中の出現順を使う．
    """
    skeletons = {}
    for label in root.findall('./meta//label'):
        name = label.findtext('name')
        parent = label.findtext('parent')
        if parent and name and name not in skeletons.setdefault(parent, []):
            skeletons[parent].append(name)
    if not skeletons:
        for skeleton in root.iter('skeleton'):
            points = skeletons.setdefault(skeleton.get('label'), [])
            for pt in skeleton.findall('points'):
                if pt.get('label') not in points:
                    points.append(pt.get('label'))
    return skeletons


def convert_cvat_to_yolo_pose(xml_file, output_dir, classes=None, keypoint_labels=None,
                              box_from_keypoints=True):
    """CVAT for images 1.1 の XML（skeleton）を YOLO-pose 形式の txt に変換する．

    classes:          クラス名のリスト（順番がクラスID）．省略時は XML のスケルトン定義順
    keypoint_labels:  キーポイントの並び順．省略時はクラスごとに XML の定義順
    box_from_keypoints: group_id で対応付く box が無いとき，見えている点を囲む矩形を使う

    使ったクラス一覧と，1インスタンスあたりのキーポイント数を返す
    （data.yaml の kpt_shape は [キーポイント数, 3]）．
    """
    # XMLファイルの読み込み
    tree = ET.parse(xml_file)
    root = tree.getroot()

    skeletons = read_skeletons(root)
    if classes is None:
        classes = list(skeletons)

    # YOLOのフォーマットではキーポイントの数と順番を全クラスで揃える必要がある
    if keypoint_labels is not None:
        layout = {cls: list(keypoint_labels) for cls in classes}
    else:
        layout = {cls: skeletons.get(cls, []) for cls in classes}
    counts = {len(v) for v in layout.values()}
    if len(counts) > 1:
        detail = ', '.join(f'{cls}={len(v)}' for cls, v in layout.items())
        raise ValueError(f'クラスごとのキーポイント数が揃っていません（{detail}）．'
                         'keypoint_labels で共通の並び順を指定してください')
    num_kpts = counts.pop() if counts else 0

    # 出力先ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # 各画像要素をループ
    for image in root.findall('image'):
        image_name = image.get('name')
        img_width = float(image.get('width'))
        img_height = float(image.get('height'))

        # 画像内のバウンディングボックスを取得し、group_idごとに整理
        boxes_by_group = {}
        for box in image.findall('box'):
            group_id = box.get('group_id')
            if group_id:
                boxes_by_group[group_id] = box

        lines = []
        for skeleton in image.findall('skeleton'):
            label = skeleton.get('label')
            if label not in classes:
                continue
            class_id = classes.index(label)

            # 1. キーポイント（スケルトン）の取得
            points_dict = {}
            for pt in skeleton.findall('points'):
                px, py = map(float, pt.get('points').split(',')[:2])

                # YOLOのVisibility定義:
                # 0: 画像外(未定義), 1: 隠蔽(occluded), 2: 見えている(visible)
                visibility = 2
                if pt.get('outside') == '1':
                    visibility = 0
                elif pt.get('occluded') == '1':
                    visibility = 1

                points_dict[pt.get('label')] = (px, py, visibility)

            # 2. バウンディングボックス（group_idで対応付いたもの，無ければ点を囲む矩形）
            box = boxes_by_group.get(skeleton.get('group_id'))
            if box is not None:
                xtl, ytl = float(box.get('xtl')), float(box.get('ytl'))
                xbr, ybr = float(box.get('xbr')), float(box.get('ybr'))
            elif box_from_keypoints:
                visible = [(x, y) for x, y, v in points_dict.values() if v > 0]
                if not visible:
                    continue
                xs, ys = zip(*visible)
                xtl, ytl, xbr, ybr = min(xs), min(ys), max(xs), max(ys)
            else:
                continue

            x_center = ((xtl + xbr) / 2) / img_width
            y_center = ((ytl + ybr) / 2) / img_height
            width = (xbr - xtl) / img_width
            height = (ybr - ytl) / img_height
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

            # 定義したキーポイントの順番に沿って文字列を結合
            for kp_label in layout[label]:
                if kp_label in points_dict and points_dict[kp_label][2] > 0:
                    px, py, vis = points_dict[kp_label]
                    line += f" {px / img_width:.6f} {py / img_height:.6f} {vis}"
                else:
                    # 該当するキーポイントがアノテーションされていない・画像外の場合
                    line += " 0.000000 0.000000 0"
            lines.append(line + '\n')

        if not lines:
            continue

        # テキストファイルの出力先パスを作成（例: images/0color.png -> output/0color.txt）
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        with open(os.path.join(output_dir, f"{base_name}.txt"), 'w') as f:
            f.writelines(lines)

    return classes, num_kpts


def read_list_file(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CVAT for images 1.1 の XML（skeleton）を YOLO-pose 形式に変換する')
    parser.add_argument('xml', nargs='?', default='annotations.xml',
                        help='CVATから出力したXMLファイル（既定: annotations.xml）')
    parser.add_argument('-o', '--output', default='yolo_pose_labels',
                        help='テキストファイルを出力するフォルダ（既定: yolo_pose_labels）')
    parser.add_argument('--classes', metavar='FILE',
                        help='classes.txt（1行1クラス．行の順番がクラスID）．省略時は XML のスケルトン定義順')
    parser.add_argument('--keypoints', metavar='FILE',
                        help='キーポイント名を1行1つ並べたファイル．省略時は XML の定義順')
    parser.add_argument('--require-box', action='store_true',
                        help='group_id で対応付く box が無いスケルトンは出力しない')
    args = parser.parse_args()

    classes, num_kpts = convert_cvat_to_yolo_pose(
        args.xml, args.output,
        classes=read_list_file(args.classes) if args.classes else None,
        keypoint_labels=read_list_file(args.keypoints) if args.keypoints else None,
        box_from_keypoints=not args.require_box)
    print(f"変換が完了しました。テキストファイルは '{args.output}' ディレクトリに出力されています。")
    print(f"data.yaml には names: {classes} / kpt_shape: [{num_kpts}, 3] を指定してください。")
