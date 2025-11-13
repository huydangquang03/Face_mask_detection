import os
import shutil
import random
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm
import yaml

def create_yolo_folders(output_path):
    os.makedirs(os.path.join(output_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'val'), exist_ok=True)
    return output_path

def convert_xml_to_yolo(xml_file, image_path, class_map):

        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Đọc kích thước ảnh
        img = cv2.imread(image_path)
        if img is None:
            print(f"Không thể đọc ảnh: {image_path}")
            return None
        
        img_height, img_width = img.shape[:2]
        
        yolo_annotations = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            
            if class_name not in class_map:
                print(f"Warning: Class '{class_name}' không có trong class_map, bỏ qua.")
                continue
                
            class_id = class_map[class_name]
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Chuyển đổi sang định dạng YOLO (x_center, y_center, width, height) 
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return yolo_annotations


def process_face_mask_dataset(dataset_path, output_path, class_map, train_ratio=0.8):
    create_yolo_folders(output_path)
    
    images_dir = os.path.join(dataset_path, 'images')
    annotations_dir = os.path.join(dataset_path, 'annotations')
    
    # Danh sách các file ảnh
    image_extensions = ['.jpg', '.jpeg', '.png']
    all_files = os.listdir(images_dir)
    image_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    print(f"Tìm thấy {len(image_files)} ảnh")
    random.shuffle(image_files)
    # Chia tập train/val
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Chia thành {len(train_files)} ảnh train và {len(val_files)} ảnh validation")
    
    # Xử lý tập train
    successful_train = process_image_set(train_files, 'train', images_dir, annotations_dir, output_path, class_map)
    # Xử lý tập val
    successful_val = process_image_set(val_files, 'val', images_dir, annotations_dir, output_path, class_map)
    
    print(f"Đã xử lý thành công {successful_train} ảnh train và {successful_val} ảnh validation")
    
    # Tạo file data.yaml
    create_data_yaml(output_path, class_map)
    
    print(f"Dataset đã được chuyển đổi và lưu tại: {output_path}")
    
    return successful_train + successful_val

def process_image_set(image_files, set_name, images_dir, annotations_dir, output_path, class_map):
    successful = 0
    
    for img_file in tqdm(image_files, desc=f"Xử lý tập {set_name}"):
        img_path = os.path.join(images_dir, img_file)
        img_name = os.path.splitext(img_file)[0]
        
        # File annotation XML 
        xml_file = os.path.join(annotations_dir, f"{img_name}.xml")
        
        # Chuyển đổi annotation XML sang định dạng YOLO
        yolo_annotations = convert_xml_to_yolo(xml_file, img_path, class_map)
        
        if yolo_annotations is None or len(yolo_annotations) == 0:
            print(f"Không có annotation hợp lệ cho {img_file}, bỏ qua.")
            continue
        
        dst_img_path = os.path.join(output_path, 'images', set_name, img_file)
        shutil.copy(img_path, dst_img_path)
        
        # Lưu file annotation YOLO
        dst_label_path = os.path.join(output_path, 'labels', set_name, f"{img_name}.txt")
        with open(dst_label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        successful += 1
    
    return successful

def create_data_yaml(output_path, class_map):
    class_names = [None] * len(class_map)
    for name, idx in class_map.items():
        class_names[idx] = name
    
    yaml_content = {
        'train': os.path.join(output_path, 'images', 'train'),
        'val': os.path.join(output_path, 'images', 'val'),
        'nc': len(class_map),
        'names': class_names
    }
    
    with open(os.path.join(output_path, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"File data.yaml đã được tạo tại {os.path.join(output_path, 'data.yaml')}")

def main():
    dataset_path = 'E:\\Project\\Dataset_mask_detection'  
    output_path = 'E:\\Project\\mask_detection_yolo'      
    
    # Class map cho face mask detection
    class_map = {
        'with_mask': 0,
        'without_mask': 1,
        'mask_weared_incorrect': 2
    }
    
    # Tỷ lệ chia train/val
    train_ratio = 0.8
    
    # Xử lý dataset
    process_face_mask_dataset(dataset_path, output_path, class_map, train_ratio)

if __name__ == "__main__":
    main()