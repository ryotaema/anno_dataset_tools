import xml.etree.ElementTree as ET
import os

def convert_cvat_to_yolo(xml_file, output_dir, classes):
    """
    CVAT for images 1.1 の XMLファイルを YOLO形式の txtファイルに変換する
    """
    # XMLファイルの読み込み
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
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
        
        has_box = False
        lines = []
        
        for box in image.findall('box'):
            label = box.get('label')
            if label not in classes:
                continue
            
            has_box = True
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
        if has_box:
            with open(txt_path, 'w') as f:
                f.writelines(lines)

# --- 実行設定 ---
if __name__ == "__main__":
    # CVATから出力したXMLファイルのパス
    XML_PATH = 'annotations.xml' 
    
    # YOLO形式のテキストファイルを出力するフォルダ
    OUTPUT_DIRECTORY = 'yolo_labels' 
    
    # データセットに存在するクラス名を順番に指定してください (0番目からIDが割り振られます)
    CLASS_LIST = ['bell_pepper'] 
    
    convert_cvat_to_yolo(XML_PATH, OUTPUT_DIRECTORY, CLASS_LIST)
    print("YOLO形式への変換が完了しました。")