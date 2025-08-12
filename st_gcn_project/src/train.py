import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from stgcn_model import create_model

# 設定 PyTorch 多執行緒 (針對 8 核心 CPU 優化)
torch.set_num_threads(8)
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

class SkeletonDataset(Dataset):
    """Dataset class for skeleton data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Trainer:
    def __init__(self, model, device, save_dir="models"):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # 混合精度訓練 (針對 RTX A2000 記憶體優化)
        self.scaler = GradScaler()
        self.use_amp = True
        
        # 梯度累積設定 (等效更大批次大小)
        self.gradient_accumulation_steps = 2
        
        os.makedirs(save_dir, exist_ok=True)
        
        # GPU 記憶體優化
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清理 GPU 記憶體
            print(f"GPU Memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch with mixed precision and gradient accumulation"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 重設優化器梯度 (在累積開始前)
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            # 混合精度前向傳播
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = criterion(output, target)
                    # 梯度累積正規化
                    loss = loss / self.gradient_accumulation_steps
                
                # 混合精度反向傳播
                self.scaler.scale(loss).backward()
            else:
                output = self.model(data)
                loss = criterion(output, target)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # 梯度累積和優化
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
                # GPU 記憶體管理 (每隔一段時間清理)
                if (batch_idx + 1) % (self.gradient_accumulation_steps * 10) == 0:
                    torch.cuda.empty_cache()
            
            # 統計 (使用原始損失)
            total_loss += loss.item() * self.gradient_accumulation_steps
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 更新進度條
            accuracy = 100. * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            
            # 顯示 GPU 記憶體使用情況
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{accuracy:.2f}%',
                    'GPU': f'{gpu_memory:.1f}GB'
                })
            else:
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """Validate the model with mixed precision"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # 驗證時也使用混合精度加速
                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                        loss = criterion(output, target)
                else:
                    output = self.model(data)
                    loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # 更新進度條 (包含 GPU 記憶體資訊)
                accuracy = 100. * correct / total
                avg_loss = total_loss / (batch_idx + 1)
                
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    pbar.set_postfix({
                        'Loss': f'{avg_loss:.4f}',
                        'Acc': f'{accuracy:.2f}%',
                        'GPU': f'{gpu_memory:.1f}GB'
                    })
                else:
                    pbar.set_postfix({
                        'Loss': f'{avg_loss:.4f}',
                        'Acc': f'{accuracy:.2f}%'
                    })
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_model(self, epoch, optimizer, is_best=False):
        """Save model checkpoint"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(state, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(state, best_path)
            print(f"New best model saved with validation accuracy: {self.best_val_acc:.2f}%")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.close()
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, 
              weight_decay=1e-4, patience=10):
        """Main training loop"""
        # Setup optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=patience//2, verbose=True
        )
        
        best_epoch = 0
        no_improve = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save model if improved
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
            
            self.save_model(epoch, optimizer, is_best)
            
            # Early stopping
            if no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {best_epoch})")
                break
        
        # Plot training history
        self.plot_training_history()
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {best_epoch})")
        print(f"Model saved to: {self.save_dir}")

def load_processed_data():
    """Load processed data"""
    processed_dir = "data/processed"
    
    # Load data
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    
    # Load metadata
    with open(os.path.join(processed_dir, "metadata.pkl"), 'rb') as f:
        metadata = pickle.load(f)
    
    return X_train, X_val, y_train, y_val, metadata

def main():
    # Check if processed data exists
    if not os.path.exists("data/processed/X_train.npy"):
        print("Processed data not found. Please run data_processor.py first.")
        return
    
    # Load data
    print("Loading processed data...")
    X_train, X_val, y_train, y_val, metadata = load_processed_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Number of classes: {metadata['num_classes']}")
    
    # Create datasets and data loaders (優化 RTX A2000 6GB VRAM)
    train_dataset = SkeletonDataset(X_train, y_train)
    val_dataset = SkeletonDataset(X_val, y_val)
    
    # 批次大小優化為 RTX A2000 6GB 顯存
    batch_size = 16  # 適合 6GB 顯存
    num_workers = 8   # 充分利用 8 核心 CPU
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,  # 加速 GPU 傳輸
        persistent_workers=True  # 減少重新創建工作程序的開销
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create model with hardware-specific optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model = create_model(
        num_classes=metadata['num_classes'],
        num_joints=len(metadata['joint_names']),
        in_channels=3,  # x, y, z coordinates
        dropout=0.5
    )
    
    model = model.to(device)
    
    # 優化模型記憶體使用
    if torch.cuda.is_available():
        # 預編譯模型以獲得更好性能 (RTX A2000 支援)
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled with torch.compile for better performance")
        except:
            print("torch.compile not available, using standard model")
    
    # Create trainer and start training
    trainer = Trainer(model, device, save_dir="models")
    
    # Training hyperparameters
    epochs = 100
    learning_rate = 0.001
    weight_decay = 1e-4
    patience = 15
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        patience=patience
    )

if __name__ == "__main__":
    main()