# Training Script for Model B: TB Classifier (Normal vs TB vs Abnormal)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
import time
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler


# CONFIGURATION
# ============================================================
BATCH_SIZE = 64  
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 6
PATIENCE = 5
RANDOM_SEED = 42
USE_AMP = True

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)



# DATASET CLASS

class TBDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['filepath'], cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"⚠️ Warning: Unable to read image at {row['filepath']}")
            img = np.zeros((224, 224), dtype=np.uint8)
        
        img = np.stack([img, img, img], axis=2)
        
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(row['label'], dtype=torch.long)
        return img, label


# ============================================================
# TRANSFORMS
# ============================================================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((224, 224)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ============================================================
# MODEL
# ============================================================
def create_model_b():
    """Create DenseNet-121 for 3-class TB classification."""
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential( # type:ignore
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 3)   # 3 classes: normal, tb, abnormal
    )
    
    return model



# TRAINING FUNCTION

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            with autocast('cuda', enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        epoch_auc = 0.0
    
    return epoch_loss, epoch_acc, epoch_f1, epoch_auc



# MAIN

def main():
    # ---- Paths ----
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../data_pr/preprocessed'))
    MASTER_CSV = os.path.join(BASE_DIR, 'model_b_dataset.csv')   
    
    MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../models'))
    RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../results'))
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # ---- Device ----
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("STARTING TRAINING - MODEL B (TB CLASSIFIER)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"Model will be saved to: {MODEL_DIR}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    
    # ---- Verify CSV ----
    if not os.path.exists(MASTER_CSV):
        raise FileNotFoundError(f"❌ Master CSV not found at: {MASTER_CSV}")
    print(f"✅ Master CSV found: {os.path.basename(MASTER_CSV)}")
    
    # ---- Load and split ----
    df = pd.read_csv(MASTER_CSV)
    
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")
    
    print(f"\nTrain class distribution:")
    print(train_df['class_name'].value_counts().to_string())
    
    print(f"\nVal class distribution:")
    print(val_df['class_name'].value_counts().to_string())
    
    print(f"\nTest class distribution:")
    print(test_df['class_name'].value_counts().to_string())
    
    # ---- Datasets ----
    train_dataset = TBDataset(train_df, transform=train_transform)
    val_dataset = TBDataset(val_df, transform=val_transform)
    test_dataset = TBDataset(test_df, transform=val_transform)
    
    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        prefetch_factor=2, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        prefetch_factor=2, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        prefetch_factor=2, persistent_workers=True
    )
    
    # ---- Model ----
    model = create_model_b().to(DEVICE)
    
    # ---- Loss, Optimizer, Scheduler, Scaler ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    scaler = GradScaler('cuda', enabled=USE_AMP)
    
    # ---- History ----
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    print(f"\nBatch Size: {BATCH_SIZE} | Workers: {NUM_WORKERS} | AMP: {USE_AMP}")
    print("=" * 70)
    
    start_time = time.time()
    
    
    # TRAINING LOOP
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss, val_acc, val_f1, val_auc = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
              f"F1: {val_f1:.4f} AUC: {val_auc:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            save_path = os.path.join(MODEL_DIR, 'best_model_b.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
            }, save_path)
            print(f"  ✅ Best model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print("=" * 70)
    
    # ---- Save history ----
    history_path = os.path.join(RESULTS_DIR, 'model_b_history.csv')
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"✅ Training history saved to: {history_path}")
    
    
    # TEST EVALUATION
    
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)
    
    # ---- Load best model ----
    checkpoint = torch.load(os.path.join(MODEL_DIR, 'best_model_b.pth'), map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loss, test_acc, test_f1, test_auc = validate(model, test_loader, criterion, DEVICE)
    
    print(f"\nTest Results:")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"  AUC-ROC:  {test_auc:.4f}")
    
    # ---- Confusion matrix & classification report ----
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with autocast('cuda', enabled=USE_AMP):
                outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    class_names = ['Normal', 'TB', 'Abnormal']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())


if __name__ == "__main__":
    main()