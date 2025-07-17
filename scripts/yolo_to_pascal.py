import os
import xml.etree.ElementTree as ET
from PIL import Image
import argparse

def load_class_list(class_file_path):
    if os.path.exists(class_file_path):
        with open(class_file_path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    else:
        print(f"Warning: Class file '{class_file_path}' not found. Using empty class list.")
        return []

def yolo_to_voc(yolo_label_path, img_path, voc_save_dir, class_list, error_log):
    img = Image.open(img_path)
    width, height = img.size

    with open(yolo_label_path, 'r') as f:
        lines = f.readlines()

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'filename').text = os.path.basename(img_path)
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'  # RGB前提

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            msg = f"Invalid label format in {yolo_label_path}: '{line.strip()}'"
            print(msg)
            error_log.append(msg)
            continue

        cls_id_str, x_center, y_center, w, h = parts
        try:
            cls_id = int(cls_id_str)
        except ValueError:
            msg = f"Invalid class ID '{cls_id_str}' in {yolo_label_path}"
            print(msg)
            error_log.append(msg)
            continue

        if cls_id < 0 or cls_id >= len(class_list):
            msg = f"Class ID {cls_id} out of range in {yolo_label_path}"
            print(msg)
            error_log.append(msg)
            continue

        xmin = int((float(x_center) - float(w) / 2) * width)
        ymin = int((float(y_center) - float(h) / 2) * height)
        xmax = int((float(x_center) + float(w) / 2) * width)
        ymax = int((float(y_center) + float(h) / 2) * height)

        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = class_list[cls_id]
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    os.makedirs(voc_save_dir, exist_ok=True)
    xml_filename = os.path.splitext(os.path.basename(img_path))[0] + '.xml'
    xml_path = os.path.join(voc_save_dir, xml_filename)

    tree = ET.ElementTree(annotation)
    tree.write(xml_path, encoding='utf-8')

    print(f"Saved VOC annotation to {xml_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO annotations to Pascal VOC XML.')
    parser.add_argument('--class_file', type=str, default='anno_dataset_tools/classes.txt', help='Path to class list file')
    parser.add_argument('--yolo_labels_dir', type=str, default='dataset/yolo_dir/labels', help='Directory of YOLO label txt files')
    parser.add_argument('--yolo_images_dir', type=str, default='dataset/yolo_dir/images', help='Directory of YOLO images')
    parser.add_argument('--voc_save_dir', type=str, default='anno_dataset_tools/dataset/pascal_dir', help='Output directory for VOC XML files')
    parser.add_argument('--error_log', type=str, default='convert_errors_v2.log', help='File path to save error logs')
    args = parser.parse_args()

    class_list = load_class_list(args.class_file)
    error_log = []

    label_files = [f for f in os.listdir(args.yolo_labels_dir) if f.endswith('.txt')]

    for label_file in label_files:
        label_path = os.path.join(args.yolo_labels_dir, label_file)
        img_filename = os.path.splitext(label_file)[0] + '.jpg'
        img_path = os.path.join(args.yolo_images_dir, img_filename)

        if not os.path.exists(img_path):
            msg = f"Image file not found for label {label_file}, skipping."
            print(msg)
            error_log.append(msg)
            continue

        yolo_to_voc(label_path, img_path, args.voc_save_dir, class_list, error_log)

    if error_log:
        with open(args.error_log, 'w') as f:
            for e in error_log:
                f.write(e + '\n')
        print(f"Errors logged to {args.error_log}")

if __name__ == '__main__':
    main()
