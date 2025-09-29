import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.transforms import Compose, ToTensor,Normalize
from torchvision.transforms import Compose, ToTensor, ColorJitter, RandomHorizontalFlip,RandomAffine

class MaskDetectionDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (str): Đường dẫn đến thư mục gốc của dataset, ví dụ: "E:/CPV_DL/Dataset_mask_detection"
            transform (callable, optional): Các phép biến đổi áp dụng cho hình ảnh.
        """
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(root, "images")
        self.annotation_dir = os.path.join(root, "annotations")
        
        # Lấy danh sách các file hình ảnh (đuôi .png)
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        
        # Tạo danh sách file annotation tương ứng (đuôi .xml)
        self.annotation_files = [f.replace('.png', '.xml') for f in self.image_files]
        
        # Danh sách các lớp (có thể điều chỉnh dựa trên dataset thực tế)
        self.categories = ["with_mask", "without_mask","mask_weared_incorrect"]
    
    def __len__(self):
        """Trả về số lượng mẫu trong dataset."""
        return len(self.image_files)
    
    def __getitem__(self, index):
        """
        Trả về một mẫu dữ liệu (image, target) tại chỉ số index.
        
        Returns:
            image (tensor): Hình ảnh đã được biến đổi.
            target (dict): Dictionary chứa 'boxes' và 'labels'.
        """
        # Lấy đường dẫn file hình ảnh và annotation
        image_path = os.path.join(self.image_dir, self.image_files[index])
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[index])
        
        # Đọc hình ảnh
        image = Image.open(image_path).convert("RGB")
        
        # Phân tích annotation
        boxes, labels = self._parse_annotation(annotation_path)
        
        # Chuyển đổi sang tensor
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        
        # Áp dụng transform cho hình ảnh nếu có
        if self.transform:
            image = self.transform(image)
        
        # Tạo target dictionary cho FasterRCNN
        target = {
            'boxes': boxes,  # Tensor [N, 4] với [xmin, ymin, xmax, ymax]
            'labels': labels  # Tensor [N] với chỉ số lớp
        }
        
        return image, target
    
    def _parse_annotation(self, annotation_path):
        """
        Phân tích file XML annotation để trích xuất bounding boxes và labels.
        
        Args:
            annotation_path (str): Đường dẫn đến file XML annotation.
        
        Returns:
            boxes (list): Danh sách các bounding box [xmin, ymin, xmax, ymax].
            labels (list): Danh sách các chỉ số lớp.
        """
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            # Trích xuất tên lớp và ánh xạ sang chỉ số
            name = obj.find('name').text
            label = self.categories.index(name) +1
            
            # Trích xuất bounding box
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        
        return boxes, labels
if __name__ == "__main__":
    transform = Compose([
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,hue=0.1),
        #ToTensor(),
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    dataset = MaskDetectionDataset(root="E:/Project/Dataset_mask_detection", transform=transform)
    image, target = dataset[34]  # Lấy mẫu đầu tiên
    #image.show()# In kích thước tensor của hình ảnh
    image.show()
    # print(image)
    # print(target)  # In target dictionary