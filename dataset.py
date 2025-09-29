import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.transforms import Compose, ToTensor,Normalize
from torchvision.transforms import Compose, ToTensor, ColorJitter, RandomHorizontalFlip,RandomAffine

class MaskDetectionDataset(Dataset):
    def __init__(self, root, transform=None):

        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(root, "images")
        self.annotation_dir = os.path.join(root, "annotations")
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        self.annotation_files = [f.replace('.png', '.xml') for f in self.image_files]
        self.categories = ["with_mask", "without_mask","mask_weared_incorrect"]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):

        image_path = os.path.join(self.image_dir, self.image_files[index])
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[index])
        image = Image.open(image_path).convert("RGB")
        boxes, labels = self._parse_annotation(annotation_path)
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        if self.transform:
            image = self.transform(image)
        target = {
            'boxes': boxes,  
            'labels': labels  
        }
        
        return image, target
    
    def _parse_annotation(self, annotation_path):

        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
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