import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn,fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def load_model(checkpoint_path, device):
    # Load the model architecture
    model = fasterrcnn_mobilenet_v3_large_fpn()
    in_feature = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feature, 4)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = Compose([ToTensor()])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor, image

def draw_predictions(image, predictions, categories, confidence_threshold, output_path):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    colors = ['r', 'g', 'b']  # Different colors for different classes
    
    for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
        if score >= confidence_threshold:
            box = box.cpu().numpy()
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Create rectangle patch
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, 
                                   edgecolor=colors[label-1], facecolor='none')
            ax.add_patch(rect)
            
            # Add label and score
            label_text = f'{categories[label-1]}: {score:.2f}'
            plt.text(x1, y1-10, label_text, color=colors[label-1], fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def test_image():
    # Define categories
    categories = ["with_mask", "without_mask", "mask_weared_incorrect"]
    
    # Set paths and parameters
    image_path ="E:\\WindowsApps\\Downloads\\testanh.jpeg" # Replace with your test image path
    checkpoint_path = 'E:\\Project\\trained_model_FasterCNN\\best.pt'
    output_path = 'E:/Project/output_prediction.jpg'
    confidence_threshold = 0.3
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Load and preprocess image
    image_tensor, original_image = preprocess_image(image_path)
    
    # Perform inference
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model([image_tensor])[0]
    
    # Draw and save predictions
    draw_predictions(original_image, predictions, categories, 
                    confidence_threshold, output_path)
    
    print(f"Prediction saved to {output_path}")
    
    # Print predictions
    print("\nDetected objects:")
    for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
        if score >= confidence_threshold:
            print(f"Class: {categories[label-1]}, Score: {score:.2f}, Box: {box.cpu().numpy()}")

if __name__ == '__main__':
    test_image()