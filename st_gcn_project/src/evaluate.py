import torch
import torch.nn as nn
from torch import amp
import numpy as np
import os
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from stgcn_model import create_model
from train import SkeletonDataset
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 設定 PyTorch 多執行緒 (針對 8 核心 CPU 優化)
torch.set_num_threads(8)
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

class Evaluator:
    def __init__(self, model, device, use_amp=True):
        self.model = model
        self.device = device
        self.use_amp = use_amp  # 混合精度推理
        
        # GPU 記憶體優化
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU Memory before evaluation: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
    def evaluate(self, test_loader, class_names=None):
        """Evaluate model on test data"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        print("Evaluating model with hardware optimizations...")
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # 使用混合精度推理加速
                if self.use_amp:
                    with amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
                        output = self.model(data)
                else:
                    output = self.model(data)
                    
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # 定期清理 GPU 記憶體
                if batch_idx % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        macro_f1 = f1_score(all_targets, all_predictions, average='macro')
        
        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        
        # Classification report
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(all_targets)))]
        
        print(f"\nClassification Report:")
        print(classification_report(
            all_targets,
            all_predictions,
            target_names=class_names,
            digits=4,
            zero_division=0
        ))
        
        # Confusion matrix analysis
        cm = confusion_matrix(all_targets, all_predictions)
        self.analyze_confusion_matrix(cm, class_names)
        
        # Save results
        self.save_results(accuracy, macro_f1, cm, all_targets, all_predictions, class_names)
        
        return accuracy, macro_f1, cm
    
    def analyze_confusion_matrix(self, cm, class_names):
        """Analyze confusion matrix and find most confused pairs"""
        print(f"\nConfusion Matrix Analysis:")
        print("=" * 60)
        
        # For each class, find the most confused class
        most_confused = {}
        confusion_pairs = []
        
        for i in range(len(class_names)):
            # Get confusion counts for class i (excluding correct predictions)
            confusion_counts = cm[i].copy()
            confusion_counts[i] = 0  # Exclude correct predictions
            
            if np.sum(confusion_counts) > 0:
                most_confused_idx = np.argmax(confusion_counts)
                most_confused_count = confusion_counts[most_confused_idx]
                most_confused[i] = (most_confused_idx, most_confused_count)
                
                print(f"{class_names[i]} is most confused with {class_names[most_confused_idx]} "
                      f"({most_confused_count} times)")
        
        # Find mutual confusion pairs
        print(f"\nMutual Confusion Pairs:")
        print("-" * 30)
        
        mutual_pairs = []
        for i in most_confused:
            confused_with, count_i = most_confused[i]
            if confused_with in most_confused:
                confused_back, count_back = most_confused[confused_with]
                if confused_back == i and i < confused_with:  # Avoid duplicates
                    mutual_pairs.append((i, confused_with, count_i, count_back))
                    print(f"{class_names[i]} <-> {class_names[confused_with]} "
                          f"({count_i} and {count_back} confusions)")
        
        if not mutual_pairs:
            print("No mutual confusion pairs found.")
        
        return most_confused, mutual_pairs
    
    def plot_confusion_matrix(self, cm, class_names, save_path="results/confusion_matrix.png"):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix plot saved to: {save_path}")
    
    def save_results(self, accuracy, macro_f1, cm, y_true, y_pred, class_names):
        """Save evaluation results"""
        os.makedirs("results", exist_ok=True)
        
        # Save metrics
        results = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'class_names': class_names
        }
        
        with open("results/evaluation_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Save confusion matrix plot
        self.plot_confusion_matrix(cm, class_names)
        
        # Save detailed report
        with open("results/evaluation_report.txt", 'w', encoding='utf-8') as f:
            f.write("ST-GCN Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Overall Metrics:\n")
            f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"Macro F1-Score: {macro_f1:.4f}\n\n")
            
            f.write("Per-Class Results:\n")
            f.write("-" * 30 + "\n")
            
            # Calculate per-class metrics
            for i, class_name in enumerate(class_names):
                class_mask = (y_true == i)
                if np.any(class_mask):
                    class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
                    class_f1 = f1_score(y_true == i, y_pred == i)
                    f.write(f"{class_name}: Accuracy={class_acc:.4f}, F1={class_f1:.4f}\n")
            
            f.write(f"\nConfusion Matrix:\n")
            f.write(str(cm))
            
            # Most confused analysis
            most_confused, mutual_pairs = self.analyze_confusion_matrix(cm, class_names)
            f.write(f"\n\nMost Confused Classes:\n")
            f.write("-" * 25 + "\n")
            for i in most_confused:
                confused_with, count = most_confused[i]
                f.write(f"{class_names[i]} -> {class_names[confused_with]} ({count} times)\n")
            
            f.write(f"\nMutual Confusion Pairs:\n")
            f.write("-" * 25 + "\n")
            for i, j, count_i, count_j in mutual_pairs:
                f.write(f"{class_names[i]} <-> {class_names[j]} ({count_i} and {count_j} confusions)\n")
        
        print(f"Evaluation results saved to: results/")

def load_model(model_path, metadata):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(
        num_classes=metadata['num_classes'],
        num_joints=len(metadata['joint_names']),
        in_channels=3,
        dropout=0.5
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded from: {model_path}")
    print(f"Best validation accuracy during training: {checkpoint.get('best_val_acc', 'N/A')}")
    
    return model, device

def load_processed_data():
    """Load processed data"""
    processed_dir = "data/processed"
    
    # Load validation data
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    
    # Load metadata
    with open(os.path.join(processed_dir, "metadata.pkl"), 'rb') as f:
        metadata = pickle.load(f)
    
    return X_val, y_val, metadata

def main():
    # Check if model exists
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first by running train.py")
        return
    
    # Check if processed data exists
    if not os.path.exists("data/processed/X_val.npy"):
        print("Processed validation data not found. Please run data_processor.py first.")
        return
    
    # Load data and metadata
    print("Loading validation data...")
    X_val, y_val, metadata = load_processed_data()
    
    print(f"Validation data shape: {X_val.shape}")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Class names: {metadata['class_names']}")
    
    # Load model
    model, device = load_model(model_path, metadata)
    
    # Create data loader (硬體優化)
    val_dataset = SkeletonDataset(X_val, y_val)
    batch_size = 16  # 適合 RTX A2000 6GB 顯存
    num_workers = 8  # 充分利用 8 核心 CPU
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Evaluate
    evaluator = Evaluator(model, device)
    accuracy, macro_f1, cm = evaluator.evaluate(val_loader, metadata['class_names'])
    
    print(f"\nEvaluation completed!")
    print(f"Final Results:")
    print(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  - Macro F1-Score: {macro_f1:.4f}")
    print(f"  - Results saved to: results/")

if __name__ == "__main__":
    main()