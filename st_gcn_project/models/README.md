# 模型儲存目錄

## 說明
此資料夾用於存放訓練好的模型檔案和相關輸出，由 `src/train.py` 自動生成。

## 自動生成檔案

### 最佳模型
- **best_model.pth**: 最佳驗證準確率的模型
  - 內容: 完整的 checkpoint（模型權重、優化器狀態、訓練歷史）
  - 用途: 模型評估和部署使用

### 訓練檢查點
- **checkpoint_epoch_*.pth**: 每個 epoch 的模型檢查點
  - 格式: `checkpoint_epoch_1.pth`, `checkpoint_epoch_2.pth`, ...
  - 內容: 完整的訓練狀態快照
  - 用途: 恢復訓練或分析訓練過程

### 訓練視覺化
- **training_history.png**: 訓練歷史圖表
  - 內容: 損失值和準確率的訓練/驗證曲線
  - 用途: 分析訓練過程和模型性能

## 模型規格
- **架構**: ST-GCN (Spatial-Temporal Graph Convolutional Network)
- **輸入格式**: `[batch_size, 3, 20, 45, 1]`
- **輸出格式**: `[batch_size, num_classes]`
- **優化**: 針對 RTX A2000 6GB VRAM 優化

## 硬體優化特性
- 混合精度訓練 (AMP)
- 梯度累積
- 記憶體優化
- 多執行緒 CPU 利用

## 注意事項
- 檢查點檔案可能很大 (數百 MB)
- 訓練完成後可刪除中間檢查點，保留 best_model.pth
- 模型載入需要對應的 metadata.pkl 檔案