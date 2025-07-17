import os
import xml.etree.ElementTree as ET
from PIL import Image
import argparse

def load_class_list(class_file_path):
    if os.path.exists(class_file_path):
        with open(class_file_path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    else:
        return []

def save_class_list(class_file_path, class_list):
    with open(class_file_path, 'w') as f:
        for cls in class_list:
            f.write(cls + '\n')

def voc_to_yolo(xml_path, img_dir, yolo_save_dir, class_list, new_classes, error_log):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text
    img_path = os.path.join(img_dir, filename)

    if not os.path.exists(img_path):
        msg = f"Image file not found: {img_path}"
        print(msg)
        error_log.append(msg)
        return

    img = Image.open(img_path)
    width, height = img.size

    yolo_lines = []

    for obj in root.findall('object'):
        cls_name = obj.find('name').text.strip()
        if cls_name not in class_list:
            if cls_name not in new_classes:
                print(f"New class detected: '{cls_name}', adding to class list.")
                new_classes.append(cls_name)
            # cls_idはまだ不明。スキップしとく
            continue
        cls_id = class_list.index(cls_name)

        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    if not yolo_lines:
        msg = f"No valid objects found in {xml_path}, skipping."
        print(msg)
        error_log.append(msg)
        return

    os.makedirs(yolo_save_dir, exist_ok=True)
    txt_filename = os.path.splitext(os.path.basename(xml_path))[0] + '.txt'
    txt_path = os.path.join(yolo_save_dir, txt_filename)

    with open(txt_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

    print(f"Saved YOLO annotation to {txt_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_dir', type=str, default='anno_dataset_tools/dataset/pascal_dir')
    parser.add_argument('--img_dir', type=str, default='dataset/yolo_dir/images')
    parser.add_argument('--yolo_save_dir', type=str, default='dataset/yolo_dir/labels')
    parser.add_argument('--class_file', type=str, default='anno_dataset_tools/classes.txt')
    parser.add_argument('--error_log', type=str, default='convert_errors.log')
    args = parser.parse_args()

    class_list = load_class_list(args.class_file)
    new_classes = []
    error_log = []

    xml_files = [f for f in os.listdir(args.voc_dir) if f.endswith('.xml')]

    for xml_file in xml_files:
        xml_path = os.path.join(args.voc_dir, xml_file)
        voc_to_yolo(xml_path, args.img_dir, args.yolo_save_dir, class_list, new_classes, error_log)

    if new_classes:
        print(f"Adding new classes to {args.class_file}: {new_classes}")
        class_list.extend(new_classes)
        save_class_list(args.class_file, class_list)

    if error_log:
        with open(args.error_log, 'w') as f:
            for line in error_log:
                f.write(line + '\n')
        print(f"Errors logged to {args.error_log}")

if __name__ == '__main__':
    main()
