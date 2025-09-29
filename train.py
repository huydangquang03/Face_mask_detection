from dataset import MaskDetectionDataset
from torchvision.transforms import ToTensor,Resize,Compose,Normalize,RandomAffine,ColorJitter
import argparse
import shutil
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn,FasterRCNN_ResNet50_FPN_Weights,fasterrcnn_mobilenet_v3_large_fpn,FasterRCNN_MobileNet_V3_Large_FPN_Weights
from pprint import pprint
from torch.utils.data import random_split, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.autonotebook import tqdm
from torchvision.transforms import Compose, ToTensor, ColorJitter, RandomHorizontalFlip
from torch.utils.data import random_split, DataLoader
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Ẩn thông báo INFO và WARNING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint
from torch.optim.lr_scheduler import StepLR


def get_args():
    parser = argparse.ArgumentParser(description='Train faster CNN model')
    parser.add_argument('--data_path','-d',type=str,default='E:/Project/Dataset_mask_detection',help='duong dan path')
    parser.add_argument('--epochs','-e',type=int,default=40,help='nhap epoch')
    parser.add_argument('--batch_size','-b',type=int,default=8,help='nhap batch_size')
    parser.add_argument('--learning_rate','-l',type=float,default=1e-3,help='nhap learning rate')
    parser.add_argument('--momentum','-m',type=float,default=0.9,help='nhap momentum')
    parser.add_argument('--log_folder','-f',type=str,default='tensorboard2',help='path to folder')
    parser.add_argument('--checkpoint_folder','-c',type=str,default='trained_model_FasterCNN',help='checkpoint')
    parser.add_argument("--saved_checkpoint", "-o", type=str, default=None, help="Continue from this checkpoint")
    args = parser.parse_args()
    return args

def collate_fn(batch):
    images,labels = zip(*batch)
    return images,labels
def train_MaskDetectionDataset(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Khai báo transform riêng cho train và test
    train_transform = Compose([
        RandomHorizontalFlip(p=0.5),
        # RandomAffine(
        #     degrees=(-15,15),
        #     translate=(0.10,0.10),
        #     scale=(0.9,1.1),
        #     shear=(5)
        # ),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,hue=0.1),
        ToTensor(),
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = Compose([
        ToTensor(),
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Tạo full dataset trước (để chia index)
    full_dataset = MaskDetectionDataset(root="E:/Project/Dataset_mask_detection")
    # Chia index train/tes
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_indices, test_indices = torch.utils.data.random_split(
        list(range(len(full_dataset))), [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # để tái hiện được kết quả
    )
   
    # Tạo 2 dataset với transform riêng
    train_dataset = torch.utils.data.Subset(
        MaskDetectionDataset(root="E:/Project/Dataset_mask_detection", transform=train_transform),
        train_indices
    )

    test_dataset = torch.utils.data.Subset(
        MaskDetectionDataset(root="E:/Project/Dataset_mask_detection", transform=test_transform),
        test_indices
    )

    # Tạo DataLoader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4,num_workers=4, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=4,num_workers=4, shuffle=False, collate_fn=collate_fn)
    
    model = fasterrcnn_mobilenet_v3_large_fpn(weights =FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    in_feature = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_feature,num_classes=4)
    #optimizer = torch.optim.SGD(params=model.parameters(),lr=args.learning_rate,momentum=args.momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    model.to(device) 
    if  os.path.exists('E:/Project/trained_model_FasterCNN/last.pt'): 
        checkpoint = torch.load('E:/Project/trained_model_FasterCNN/last.pt')
        start_epoch = checkpoint["epoch"]
        best_map = checkpoint["map"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        start_epoch = 0
        best_map = -1
  
    if os.path.isdir(args.log_folder):
        shutil.rmtree(args.log_folder)
    os.makedirs(args.log_folder)
    writer = SummaryWriter(args.log_folder)
    if not os.path.isdir(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder) 
    #TRAIN
    num_iter = len(train_dataloader)
    
    for epoch in  range(start_epoch,args.epochs):
        model.train()
        progress = tqdm(train_dataloader,colour='cyan')
        train_loss = []
        for i,(images,labels) in enumerate(progress):
            images = [image.to(device) for image in images]
            labels = [{'boxes':label['boxes'].to(device),'labels':label["labels"].to(device)} for label in labels ]
            #forward
            loss = model(images,labels)
            sum_loss = sum([loss for loss in loss.values()])
            train_loss.append(sum_loss.item())
            mean_loss = np.mean(train_loss)
            progress.set_description(('Epoch{}/{}. Iteration{}/{}.Loss{:.3f}'.format(epoch+1,args.epochs,i+1,num_iter,mean_loss)))
            writer.add_scalar('train/loss', mean_loss,epoch*num_iter+i)
            #print(sum_loss)
            #backward
            optimizer.zero_grad()
            sum_loss.backward()
            optimizer.step()
            #print(sum_loss.item())
        scheduler.step()
    #VALIDATION
        model.eval()
        progress = tqdm(test_dataloader,colour='cyan')
        metric = MeanAveragePrecision(iou_type='bbox')
        for i,(images,labels) in enumerate(progress):
            images = [image.to(device) for image in images]
            with torch.no_grad():
                out_puts = model(images)
            pred = []
            for output in out_puts:
                pred.append({
                    'boxes': output['boxes'].to('cpu'),
                    'labels':output['labels'].to('cpu'),
                    'scores':output['scores'].to('cpu')
                })
            target = []
            for label in labels:
                target.append({
                    'boxes': label['boxes'].to('cpu'),
                    'labels':label['labels'].to('cpu'),
                    
                })
            metric.update(pred,target)
        result = metric.compute()
        pprint(result)
        writer.add_scalar("Val/mAP", result["map"], epoch)
        writer.add_scalar("Val/mAP_50", result["map_50"], epoch)
        writer.add_scalar("Val/mAP_75", result["map_75"], epoch)
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "map": result["map"],
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(checkpoint,"E:\\Project\\trained_model_FasterCNN\\last.pt" )
        if result["map"] > best_map:
            best_map = result["map"]
            torch.save(checkpoint,'E:\\Project\\trained_model_FasterCNN\\best.pt' )


if __name__ == '__main__':
   args = get_args()
   train_MaskDetectionDataset(args)
   
   