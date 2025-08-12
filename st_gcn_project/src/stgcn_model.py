import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphConvolution(nn.Module):
    """Graph convolution layer with adjacency matrix"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, bias=True):
        super(GraphConvolution, self).__init__()
        
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, 1, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, A):
        # x: [N, C, T, V]
        # A: [K, V, V] - adjacency matrices
        
        x = self.conv(x)  # [N, C*K, T, V]
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        
        # Apply graph convolution
        A = A.to(device=x.device, dtype=x.dtype)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        
        return self.dropout(x)

class TemporalConvolution(nn.Module):
    """Temporal convolution layer"""
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dropout=0):
        super(TemporalConvolution, self).__init__()
        
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), 
                             (stride, 1), (padding, 0))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.conv(x))

class STGCNBlock(nn.Module):
    """ST-GCN block with gating mechanism"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 dropout=0, residual=True):
        super(STGCNBlock, self).__init__()
        
        self.gcn = GraphConvolution(in_channels, out_channels, kernel_size, dropout=dropout)
        self.tcn = TemporalConvolution(out_channels, out_channels, stride=stride, dropout=dropout)
        
        # Gating mechanism
        self.gate_conv = nn.Conv2d(out_channels, out_channels, 1)
        
        # Group normalization
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.GroupNorm(min(32, out_channels), out_channels)
            )
        
    def forward(self, x, A, mask=None):
        res = self.residual(x)
        
        # Graph convolution
        x = self.gcn(x, A)
        
        # Temporal convolution
        x = self.tcn(x)
        
        # Apply mask (gating mechanism)
        if mask is not None:
            # Align mask temporal dimension to match x after temporal stride
            # mask: [N, T, V] -> align to [N, x_T, x_V]
            n, c, x_t, x_v = x.shape
            mask_aligned = mask
            # Ensure on same device/dtype
            mask_aligned = mask_aligned.to(device=x.device, dtype=x.dtype)
            # Resize if T or V mismatch
            if mask_aligned.dim() == 3 and (mask_aligned.size(1) != x_t or mask_aligned.size(2) != x_v):
                mask_aligned = F.interpolate(
                    mask_aligned.unsqueeze(1),  # [N,1,T,V]
                    size=(x_t, x_v),
                    mode='nearest'
                ).squeeze(1)  # [N, x_T, x_V]
            # Broadcast to channel dimension
            x = x * mask_aligned.unsqueeze(1)
        
        # Additional gating
        gate = torch.sigmoid(self.gate_conv(x))
        x = x * gate
        
        # Group normalization
        x = self.norm(x)
        
        # Residual connection
        x = x + res
        x = self.relu(x)
        
        return x

class WeightedPooling(nn.Module):
    """Weighted global average pooling"""
    def __init__(self, channels):
        super(WeightedPooling, self).__init__()
        self.weight_conv = nn.Conv2d(channels, 1, 1)
        
    def forward(self, x, mask=None):
        # x: [N, C, T, V]
        weights = torch.sigmoid(self.weight_conv(x))  # [N, 1, T, V]
        
        if mask is not None:
            # Apply mask to weights
            n, _, t, v = weights.shape
            mask_aligned = mask
            mask_aligned = mask_aligned.to(device=x.device, dtype=x.dtype)
            if mask_aligned.dim() == 3 and (mask_aligned.size(1) != t or mask_aligned.size(2) != v):
                mask_aligned = F.interpolate(
                    mask_aligned.unsqueeze(1),  # [N,1,T,V]
                    size=(t, v),
                    mode='nearest'
                ).squeeze(1)  # [N, t, v]
            weights = weights * mask_aligned.unsqueeze(1)
            
        # Weighted average pooling
        weighted_x = x * weights
        
        # Global average pooling with normalization
        pooled = torch.sum(weighted_x, dim=(2, 3)) / (torch.sum(weights, dim=(2, 3)) + 1e-8)
        
        return pooled

class STGCN(nn.Module):
    """ST-GCN model for skeleton-based action recognition"""
    def __init__(self, in_channels=3, num_classes=10, num_joints=13, 
                 dropout=0.5, joint_dropout=0.1, temporal_dropout=0.1):
        super(STGCN, self).__init__()
        
        self.num_joints = num_joints
        self.joint_dropout = joint_dropout
        self.temporal_dropout = temporal_dropout
        
        # Create adjacency matrix (simplified - identity + neighboring connections)
        self.register_buffer('A', self.get_adjacency_matrix(num_joints))
        
        # ST-GCN layers
        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(in_channels, 64, 3, residual=False, dropout=dropout),
            STGCNBlock(64, 64, 3, dropout=dropout),
            STGCNBlock(64, 64, 3, dropout=dropout),
            STGCNBlock(64, 128, 3, stride=2, dropout=dropout),
            STGCNBlock(128, 128, 3, dropout=dropout),
            STGCNBlock(128, 128, 3, dropout=dropout),
            STGCNBlock(128, 256, 3, stride=2, dropout=dropout),
            STGCNBlock(256, 256, 3, dropout=dropout),
            STGCNBlock(256, 256, 3, dropout=dropout),
        ])
        
        # Weighted pooling
        self.pooling = WeightedPooling(256)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def get_adjacency_matrix(self, num_joints, max_hop=2):
        """Create adjacency matrix for skeleton graph"""
        # Simplified adjacency matrix - you may want to customize this based on your skeleton structure
        A = np.eye(num_joints, dtype=np.float32)
        
        # Add connections between adjacent joints (this is a simplified version)
        for i in range(num_joints - 1):
            A[i, i + 1] = 1
            A[i + 1, i] = 1
        
        # Normalize
        A = A / np.sum(A, axis=1, keepdims=True)
        
        # Create multiple adjacency matrices for different relationships
        A_stack = np.stack([A, A, A])  # 3 different relationship types
        
        return torch.from_numpy(A_stack)
    
    def create_mask_from_input(self, x):
        """Create mask from input data (detect valid joints)"""
        # x: [N, C, T, V, M]
        # Create mask where all coordinates are non-zero
        mask = torch.sum(torch.abs(x), dim=(1, 4)) > 0  # [N, T, V]
        return mask.float()
    
    def apply_joint_dropout(self, x, mask, training=True):
        """Apply joint dropout during training"""
        if training and self.joint_dropout > 0:
            # Randomly drop some joints
            joint_keep_prob = 1 - self.joint_dropout
            joint_mask = torch.bernoulli(torch.ones_like(mask) * joint_keep_prob)
            mask = mask * joint_mask
        return mask
    
    def apply_temporal_dropout(self, x, mask, training=True):
        """Apply temporal dropout during training"""
        if training and self.temporal_dropout > 0:
            # Randomly drop some temporal frames
            temp_keep_prob = 1 - self.temporal_dropout
            temp_mask = torch.bernoulli(torch.ones_like(mask) * temp_keep_prob)
            mask = mask * temp_mask
        return mask
    
    def forward(self, x):
        # x: [N, C, T, V, M] -> [N, C, T, V] (assume M=1)
        if x.dim() == 5:
            x = x.squeeze(-1)
        
        # Create mask from input
        mask = self.create_mask_from_input(x.unsqueeze(-1))
        # Ensure mask is on same device/dtype as x for subsequent ops
        mask = mask.to(device=x.device, dtype=x.dtype)
        
        # Apply dropout to mask during training
        mask = self.apply_joint_dropout(x, mask, self.training)
        mask = self.apply_temporal_dropout(x, mask, self.training)
        
        # Apply ST-GCN blocks
        # 為避免過度抑制訊號，訓練階段不將 mask 作用於每個區塊，改僅在池化階段使用
        for st_gcn in self.st_gcn_networks:
            x = st_gcn(x, self.A, None)
        
        # Weighted pooling
        x = self.pooling(x, mask)
        
        # Classification
        x = self.classifier(x)
        
        return x

def create_model(num_classes, num_joints=13, in_channels=3, dropout=0.5):
    """Create ST-GCN model"""
    model = STGCN(
        in_channels=in_channels,
        num_classes=num_classes,
        num_joints=num_joints,
        dropout=dropout,
        joint_dropout=0.1,
        temporal_dropout=0.1
    )
    return model