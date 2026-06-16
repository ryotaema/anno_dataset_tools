import xml.etree.ElementTree as ET
import os

def convert_cvat_to_yolo_pose(xml_file, output_dir):
    # XMLファイルの読み込み
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 出力先ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # YOLOのフォーマットではキーポイントの順番を固定する必要があります。
    # XMLのメタデータに記載されていたキーポイントのラベルを順番に定義します。
    # ※必要に応じて順番を変更してください。
    keypoint_labels = ["high_peduncle", "root_peduncle", "bottom_fruit"]
    
    # クラスIDの設定（例: bell_pepperを0とする）
    class_id = 0
    
    # 各画像要素をループ
    for image in root.findall('image'):
        image_name = image.get('name')
        img_width = float(image.get('width'))
        img_height = float(image.get('height'))
        
        boxes_by_group = {}
        skeletons_by_group = {}
        
        # 画像内のバウンディングボックスを取得し、group_idごとに整理
        for box in image.findall('box'):
            group_id = box.get('group_id')
            if group_id:
                boxes_by_group[group_id] = box
                
        # 画像内のスケルトンを取得し、group_idごとに整理
        for skeleton in image.findall('skeleton'):
            group_id = skeleton.get('group_id')
            if group_id:
                skeletons_by_group[group_id] = skeleton
                
        # バウンディングボックスとスケルトンの両方が存在するgroup_idを抽出
        common_groups = set(boxes_by_group.keys()).intersection(set(skeletons_by_group.keys()))
        
        # グループ化されたboxとskeletonがない画像はスキップ
        if not common_groups:
            continue
            
        # テキストファイルの出力先パスを作成（例: images/0color.png -> output/0color.txt）
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        txt_filepath = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(txt_filepath, 'w') as f:
            for group_id in common_groups:
                box = boxes_by_group[group_id]
                skeleton = skeletons_by_group[group_id]
                
                # 1. バウンディングボックスの計算（YOLO形式に正規化）
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                
                x_center = ((xtl + xbr) / 2) / img_width
                y_center = ((ytl + ybr) / 2) / img_height
                width = (xbr - xtl) / img_width
                height = (ybr - ytl) / img_height
                
                # ボックス情報の書き出し文字列
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                
                # 2. キーポイント（スケルトン）の取得
                points_dict = {}
                for pt in skeleton.findall('points'):
                    label = pt.get('label')
                    points_str = pt.get('points')
                    occluded = pt.get('occluded')
                    outside = pt.get('outside')
                    
                    px, py = map(float, points_str.split(','))
                    
                    # YOLOのVisibility定義: 
                    # 0: 画像外(未定義), 1: 隠蔽(occluded), 2: 見えている(visible)
                    visibility = 2
                    if outside == '1':
                        visibility = 0
                    elif occluded == '1':
                        visibility = 1
                        
                    # 座標の正規化
                    px_norm = px / img_width
                    py_norm = py / img_height
                    
                    points_dict[label] = (px_norm, py_norm, visibility)
                
                # 定義したキーポイントの順番に沿って文字列を結合
                for kp_label in keypoint_labels:
                    if kp_label in points_dict:
                        px, py, vis = points_dict[kp_label]
                        line += f" {px:.6f} {py:.6f} {vis}"
                    else:
                        # 該当するキーポイントがアノテーションされていない場合
                        line += " 0.000000 0.000000 0"
                        
                # 1行として書き込み
                f.write(line + '\n')

if __name__ == "__main__":
    # 添付されたXMLファイルのパスを指定してください
    INPUT_XML = "annotations.xml"
    # テキストファイルを出力するフォルダ名
    OUTPUT_DIR = "yolo_pose_labels"
    
    convert_cvat_to_yolo_pose(INPUT_XML, OUTPUT_DIR)
    print(f"変換が完了しました。テキストファイルは '{OUTPUT_DIR}' ディレクトリに出力されています。")