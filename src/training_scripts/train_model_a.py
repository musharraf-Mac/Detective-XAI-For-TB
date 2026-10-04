# Training Script for Model A : Chest X-ray Detector (Chest X-ray vs. Not Chest X-ray)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score,f1_score, roc_auc_score
import time
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler


BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 6
PATIENCE = 5  # Early stopping patience 
RANDOM_SEED = 42
USE_AMP = True  # Use Automatic Mixed Precision for faster training

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Dataset Class

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image as grayscale
        img = cv2.imread(row['filepath'], cv2.IMREAD_GRAYSCALE)
        
        # Handle missing images
        if img is None:
            print(f"⚠️ Warning: Could not load {row['filepath']}")
            img = np.zeros((224, 224), dtype=np.uint8)
        
        # Convert grayscale to 3-channel (DenseNet expects 3 channels)
        img = np.stack([img, img, img], axis=2)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(row['label'], dtype=torch.long)
        return img, label

# Transforms

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
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

# Densnet-121 binary classifier

def create_model_a():
    """Create DenseNet-121 for binary chest X-ray detection."""
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    
    # Replace classifier for binary classification
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential( # type: ignore
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)  # 2 classes: not_chest_xray, chest_xray
    )
    
    return model

# Traning function

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
            
        # Scaler backward pass
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
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of chest X-ray
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='binary')
    epoch_auc = roc_auc_score(all_labels, all_probs)
    
    return epoch_loss, epoch_acc, epoch_f1, epoch_auc

# Training Loop

def main():
    Train_dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_pr/preprocessed/CXR_Detect_data_a/model_a_train.csv'))
    Test_dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_pr/preprocessed/CXR_Detect_data_a/model_a_test.csv'))
    Val_dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_pr/preprocessed/CXR_Detect_data_a/model_a_val.csv'))
    
    # Configuration
    
    # Create models directory if it doesn't exist
    MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Create results directory if it doesn't exist
    RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    # Check GPU memory
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = ChestXrayDataset(Train_dataset, transform=train_transform)
    val_dataset = ChestXrayDataset(Val_dataset, transform=val_transform)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Model
    model = create_model_a().to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Scheduler: Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler('cuda', enabled=USE_AMP)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING - MODEL A (CHEST X-RAY DETECTOR)")
    print(f"Batch Size: {BATCH_SIZE} | Workers: {NUM_WORKERS} | AMP: {USE_AMP}")
    print("=" * 70)
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        
        # Validate
        val_loss, val_acc, val_f1, val_auc = validate(model, val_loader, criterion, DEVICE)
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
              f"F1: {val_f1:.4f} AUC: {val_auc:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
            }, os.path.join(MODEL_DIR, 'best_model_a.pth'))
            print(f"  ✅ Best model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print("=" * 70)
    
    # Save training history
    history_path = os.path.join(RESULTS_DIR, 'model_a_history.csv')
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"✅ Training history saved to '{history_path}'")


if __name__ == "__main__":
    print("Loading datasets...")
    main()

