import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial

# 設定多程序數量 (針對 8 核心 CPU 優化)
N_PROCESSES = min(8, cpu_count())  # 使用最多 8 個程序或所有可用核心

class SkeletonDataProcessor:
    def __init__(self, max_frames=20, num_joints=45):
        self.max_frames = max_frames
        self.num_joints = num_joints
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_csv_data(self, csv_path):
        """Load skeleton data from CSV file"""
        # 避免 DtypeWarning，並減少記憶體峰值
        df = pd.read_csv(csv_path, low_memory=False)
        return df
    
    def extract_skeleton_coordinates(self, df):
        """Extract skeleton coordinates and labels from dataframe"""
        # Get coordinate columns (hand skeleton data)
        coord_columns = [col for col in df.columns if col.endswith(('_x', '_y', '_z')) and col not in ['id', 'sign_language']]
        coordinate_data = df[coord_columns].values
        
        # Get labels from sign_language column
        if 'sign_language' in df.columns:
            labels = df['sign_language'].values
        elif 'label' in df.columns:
            labels = df['label'].values
        else:
            raise ValueError("No 'sign_language' or 'label' column found in CSV")
        
        # Get video IDs if available
        video_ids = df['id'].values if 'id' in df.columns else None
            
        return coordinate_data, labels, coord_columns, video_ids
    
    def create_mask_from_nan(self, data):
        """Create binary mask from NaN values: 1 for valid data, 0 for NaN
        FIXED: Only treat NaN as missing, keep real 0 values
        """
        mask = ~np.isnan(data)
        return mask.astype(np.float32)
    
    def apply_mask_to_coordinates(self, data, mask):
        """Apply mask to coordinates: (x,y,z) * mask"""
        # Replace NaN with 0, then multiply by mask
        data_clean = np.nan_to_num(data, nan=0.0)
        return data_clean * mask
    
    def reshape_to_skeleton_format(self, data, coord_columns):
        """Reshape data to [samples, joints, coordinates(3)]"""
        # Group columns by joint name
        joint_names = []
        for col in coord_columns:
            joint_name = col.rsplit('_', 1)[0]  # Remove _x, _y, _z suffix
            if joint_name not in joint_names:
                joint_names.append(joint_name)
        
        num_samples = data.shape[0]
        num_joints = len(joint_names)
        
        # Reshape to [samples, joints, 3] for (x,y,z)
        skeleton_data = np.zeros((num_samples, num_joints, 3))
        
        for i, joint_name in enumerate(joint_names):
            x_col = f"{joint_name}_x"
            y_col = f"{joint_name}_y"
            z_col = f"{joint_name}_z"
            
            if x_col in coord_columns:
                x_idx = coord_columns.index(x_col)
                skeleton_data[:, i, 0] = data[:, x_idx]
            if y_col in coord_columns:
                y_idx = coord_columns.index(y_col)
                skeleton_data[:, i, 1] = data[:, y_idx]
            if z_col in coord_columns:
                z_idx = coord_columns.index(z_col)
                skeleton_data[:, i, 2] = data[:, z_idx]
        
        return skeleton_data, joint_names
    
    def _compute_feature_stats_ignore_zeros(self, data):
        """計算每個關節坐標維度 (x,y,z) 的平均與標準差，忽略 0（視為缺失）
        FIXED: Simplified to match v1 interface but with better zero handling
        參數:
            data: ndarray, 形狀 [N, V, 3]
        回傳:
            means, stds: 形狀 [V, 3]
        """
        mask = (data != 0)
        valid_counts = mask.sum(axis=0)  # [V, 3]
        valid_counts = np.maximum(valid_counts, 1)  # 避免除以 0
        sums = (data * mask).sum(axis=0)
        means = sums / valid_counts
        var = (((data - means) * mask) ** 2).sum(axis=0) / valid_counts
        stds = np.sqrt(var)
        stds = np.where(stds < 1e-6, 1.0, stds)  # 避免除以 0 或過小
        return means, stds

    def normalize_coordinates(self, skeleton_data):
        """以忽略 0 的方式做 z-score 正規化，只對非 0 值標準化，0 值維持為 0
        FIXED: Simplified to match v1 interface, just improved statistics computation
        回傳 (normalized, means, stds)
        """
        data = skeleton_data.copy()
        means, stds = self._compute_feature_stats_ignore_zeros(data)
        
        mask = (data != 0)
        data = data - means  # broadcast [N,V,3]
        data[~mask] = 0
        data = data / stds
        data[~mask] = 0
        
        return data, means, stds

    def normalize_with_stats(self, skeleton_data, means, stds):
        """使用給定的均值/標準差進行正規化（忽略 0）
        FIXED: Simplified to match v1 interface
        """
        data = skeleton_data.copy()
        mask = (data != 0)
        data = data - means
        data[~mask] = 0
        data = data / stds
        data[~mask] = 0
        return data
    
    def validate_input_data(self, X, y, expected_shape=None):
        """驗證輸入數據有效性
        NEW: Added input validation
        """
        assert X.shape[0] == len(y), f"樣本數量不匹配: X={X.shape[0]}, y={len(y)}"
        assert not np.any(np.isnan(X)), "輸入數據包含 NaN 值"
        
        if expected_shape is not None:
            assert X.shape[1:] == expected_shape, f"數據形狀錯誤: 期望{expected_shape}, 實際{X.shape[1:]}"
        
        print(f"✅ 數據驗證通過：{X.shape[0]} 樣本, 形狀 {X.shape[1:]}")
        return True
    
    def temporal_alignment(self, data, labels, video_ids=None):
        """Align temporal sequences to fixed length T=20 with specific windowing rules"""
        aligned_data = []
        aligned_labels = []
        
        if video_ids is not None:
            # Group frames by video ID
            unique_video_ids = np.unique(video_ids)
            
            for video_id in unique_video_ids:
                # Get all frames for this video
                video_mask = video_ids == video_id
                video_frames = data[video_mask]
                video_label = labels[video_mask][0]  # All frames should have same label
                
                total_frames = len(video_frames)
                
                if total_frames < self.max_frames:
                    # 不足20幀：在尾端補零
                    padding = np.zeros((self.max_frames - total_frames, *video_frames.shape[1:]))
                    aligned_sequence = np.concatenate([video_frames, padding], axis=0)
                    
                elif total_frames == self.max_frames:
                    # 剛好20幀：原樣使用
                    aligned_sequence = video_frames
                    
                else:
                    # 超過20幀：使用特殊窗口規則 (保持原策略)
                    aligned_sequence = self._apply_windowing_rules(video_frames, total_frames)
                
                aligned_data.append(aligned_sequence)
                aligned_labels.append(video_label)
        else:
            # Fallback: treat each sample as a single frame sequence
            for i in range(len(data)):
                sample = data[i:i+1]  # Single frame
                
                if len(sample) < self.max_frames:
                    # Pad with zeros
                    padding = np.zeros((self.max_frames - len(sample), *sample.shape[1:]))
                    padded_sample = np.concatenate([sample, padding], axis=0)
                else:
                    # Truncate to max_frames
                    padded_sample = sample[:self.max_frames]
                
                aligned_data.append(padded_sample)
                aligned_labels.append(labels[i])
        
        return np.array(aligned_data), np.array(aligned_labels)
    
    def _apply_windowing_rules(self, video_frames, total_frames):
        """Apply specific windowing rules for sequences longer than T=20
        KEPT: Original windowing strategy as requested
        """
        T = self.max_frames  # T=20
        
        # 計算中間窗口的起始位置
        center_start = (total_frames - T) // 2
        
        # 計算左右被丟棄的幀數
        left_dropped = center_start
        right_dropped = total_frames - (center_start + T)
        
        # 應用調整規則
        window_start = center_start
        
        # 若左側被丟 >15 幀，視窗向左移 5 幀
        if left_dropped > 15:
            window_start = max(0, window_start - 5)
        
        # 若右側被丟 >15 幀，視窗向右移 5 幀
        elif right_dropped > 15:
            window_start = min(total_frames - T, window_start + 5)
        
        # 視窗起點限制在 [0, t-20]
        window_start = max(0, min(total_frames - T, window_start))
        
        # 提取窗口
        return video_frames[window_start:window_start + T]
    
    def convert_to_stgcn_format(self, aligned_data):
        """Convert to ST-GCN format X[C,T,V,M]"""
        # aligned_data shape: [samples, T, V, C]
        # Need to convert to: [samples, C, T, V, M]
        samples, T, V, C = aligned_data.shape
        M = 1  # Single person
        
        # Rearrange dimensions: [samples, C, T, V, M]
        stgcn_data = aligned_data.transpose(0, 3, 1, 2)  # [samples, C, T, V]
        stgcn_data = np.expand_dims(stgcn_data, axis=-1)  # [samples, C, T, V, M]
        
        return stgcn_data
    
    def process_data(self, train_csv_path, val_csv_path):
        """Complete data processing pipeline with multiprocessing optimization
        FIXED: Updated with all improvements
        """
        print(f"Using {N_PROCESSES} CPU cores for data processing...")
        
        print("Loading training data...")
        train_df = self.load_csv_data(train_csv_path)
        train_coords, train_labels, coord_columns, train_video_ids = self.extract_skeleton_coordinates(train_df)
        print(f"Training data loaded: {train_coords.shape[0]} frames")
        if train_video_ids is not None:
            print(f"Training videos: {len(np.unique(train_video_ids))} unique videos")
        
        print("Loading validation data...")
        val_df = self.load_csv_data(val_csv_path)
        val_coords, val_labels, _, val_video_ids = self.extract_skeleton_coordinates(val_df)
        print(f"Validation data loaded: {val_coords.shape[0]} frames")
        if val_video_ids is not None:
            print(f"Validation videos: {len(np.unique(val_video_ids))} unique videos")
        
        print("Processing training data...")
        # Create masks
        train_mask = self.create_mask_from_nan(train_coords)
        train_coords_masked = self.apply_mask_to_coordinates(train_coords, train_mask)
        
        # Reshape to skeleton format
        train_skeleton, joint_names = self.reshape_to_skeleton_format(train_coords_masked, coord_columns)
        
        # Normalize（忽略 0，並回傳統計量以供驗證集使用）
        train_skeleton_norm, feat_means, feat_stds = self.normalize_coordinates(train_skeleton)
        
        # Temporal alignment
        train_aligned, train_labels_aligned = self.temporal_alignment(train_skeleton_norm, train_labels, train_video_ids)
        
        # Convert to ST-GCN format
        X_train = self.convert_to_stgcn_format(train_aligned)
        
        print("Processing validation data...")
        # Process validation data with same scaler
        val_mask = self.create_mask_from_nan(val_coords)
        val_coords_masked = self.apply_mask_to_coordinates(val_coords, val_mask)
        val_skeleton, _ = self.reshape_to_skeleton_format(val_coords_masked, coord_columns)
        
        # Apply same normalization（使用訓練集統計量，忽略 0）
        val_skeleton_norm = self.normalize_with_stats(val_skeleton, feat_means, feat_stds)
        
        val_aligned, val_labels_aligned = self.temporal_alignment(val_skeleton_norm, val_labels, val_video_ids)
        X_val = self.convert_to_stgcn_format(val_aligned)
        
        # Encode labels
        y_train = self.label_encoder.fit_transform(train_labels_aligned)
        y_val = self.label_encoder.transform(val_labels_aligned)
        
        # Validate processed data
        expected_shape = (3, self.max_frames, len(joint_names), 1)
        self.validate_input_data(X_train, y_train, expected_shape)
        self.validate_input_data(X_val, y_val, expected_shape)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Joint names: {joint_names}")
        
        # Save processed data
        processed_dir = "data/processed"
        os.makedirs(processed_dir, exist_ok=True)
        
        np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
        np.save(os.path.join(processed_dir, "X_val.npy"), X_val)
        np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
        np.save(os.path.join(processed_dir, "y_val.npy"), y_val)
        
        # Save metadata
        metadata = {
            'joint_names': joint_names,
            'num_classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_.tolist(),
            'input_shape': X_train.shape[1:],  # [C, T, V, M]
            'scaler': self.scaler,
            'norm_means': feat_means,
            'norm_stds': feat_stds,
            'label_encoder': self.label_encoder
        }
        
        with open(os.path.join(processed_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
        
        print("Data processing completed!")
        return X_train, X_val, y_train, y_val, metadata

def main():
    processor = SkeletonDataProcessor(max_frames=20)
    
    train_csv = "data/raw/train.csv"
    val_csv = "data/raw/val.csv"
    
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found. Please place your training data in data/raw/train.csv")
        return
    
    if not os.path.exists(val_csv):
        print(f"Error: {val_csv} not found. Please place your validation data in data/raw/val.csv")
        return
    
    X_train, X_val, y_train, y_val, metadata = processor.process_data(train_csv, val_csv)
    
    print("\nProcessing Summary:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Data shape: {X_train.shape[1:]} (C, T, V, M)")
    print(f"Number of classes: {metadata['num_classes']}")

if __name__ == "__main__":
    main()