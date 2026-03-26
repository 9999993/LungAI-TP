import os, sys, time, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from model import LungAIDualTaskModel, DualTaskLoss

class LungPathologyDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.labels_df = pd.read_csv(labels_file)
    def __len__(self):
        return len(self.labels_df)
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['image_path'])
        subtype_map = {'lung_aca': 0, 'lung_scc': 1, 'lung_n': 2}
        subtype = subtype_map.get(row['subtype'], 0)
        response = int(row['response'])
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(subtype, dtype=torch.long), torch.tensor(response, dtype=torch.long)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print('='*60)
print('LungAI-TP 快速训练 (小数据集)')
print('='*60)

train_dataset = LungPathologyDataset(config.TRAIN_DIR, 'data/small_train_labels.csv', transform)
val_dataset = LungPathologyDataset(config.VAL_DIR, 'data/small_val_labels.csv', transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

print(f'训练集: {len(train_dataset)} 样本')
print(f'验证集: {len(val_dataset)} 样本')

model = LungAIDualTaskModel().to(config.DEVICE)
criterion = DualTaskLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f'设备: {config.DEVICE}')
print('='*60)

best_acc = 0.0
for epoch in range(3):
    print(f'Epoch {epoch+1}/3')
    model.train()
    train_loss = 0
    train_preds, train_targets = [], []
    
    for images, subtype_labels, response_labels in tqdm(train_loader, desc='训练'):
        images = images.to(config.DEVICE)
        subtype_labels = subtype_labels.to(config.DEVICE)
        response_labels = response_labels.to(config.DEVICE)
        
        subtype_pred, response_pred = model(images)
        loss, _, _ = criterion(subtype_pred, subtype_labels, response_pred, response_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_preds.extend(torch.argmax(subtype_pred, 1).cpu().numpy())
        train_targets.extend(subtype_labels.cpu().numpy())
    
    train_acc = accuracy_score(train_targets, train_preds)
    
    # 验证
    model.eval()
    val_loss = 0
    val_preds, val_targets = [], []
    
    with torch.no_grad():
        for images, subtype_labels, response_labels in tqdm(val_loader, desc='验证'):
            images = images.to(config.DEVICE)
            subtype_labels = subtype_labels.to(config.DEVICE)
            response_labels = response_labels.to(config.DEVICE)
            
            subtype_pred, response_pred = model(images)
            loss, _, _ = criterion(subtype_pred, subtype_labels, response_pred, response_labels)
            
            val_loss += loss.item()
            val_preds.extend(torch.argmax(subtype_pred, 1).cpu().numpy())
            val_targets.extend(subtype_labels.cpu().numpy())
    
    val_acc = accuracy_score(val_targets, val_preds)
    
    print(f'训练 - 损失: {train_loss/len(train_loader):.4f}, 准确率: {train_acc:.4f}')
    print(f'验证 - 损失: {val_loss/len(val_loader):.4f}, 准确率: {val_acc:.4f}')
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), config.MODEL_PATH)
        print(f'保存最佳模型 (准确率: {best_acc:.4f})')

print('='*60)
print(f'训练完成! 最佳准确率: {best_acc:.4f}')
