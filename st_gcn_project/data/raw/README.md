# 原始資料目錄

## 說明
此資料夾用於存放原始的 CSV 資料檔案。

## 需要的檔案
請將以下檔案放入此目錄：

### 訓練資料
- **檔案名稱**: `train.csv`
- **格式**: CSV 格式，包含 id, sign_language 和骨架座標
- **內容**: 訓練用的手語影片骨架資料

### 驗證資料  
- **檔案名稱**: `val.csv`
- **格式**: CSV 格式，包含 id, sign_language 和骨架座標
- **內容**: 驗證用的手語影片骨架資料

## CSV 格式要求
```
id,sign_language,nose_x,nose_y,nose_z,left_shoulder_x,left_shoulder_y,left_shoulder_z,
right_shoulder_x,right_shoulder_y,right_shoulder_z,left_hand_0_x,left_hand_0_y,left_hand_0_z,
...left_hand_20_x,left_hand_20_y,left_hand_20_z,right_hand_0_x,right_hand_0_y,right_hand_0_z,
...right_hand_20_x,right_hand_20_y,right_hand_20_z
```

## 注意事項
- 確保 CSV 檔案編碼為 UTF-8
- 允許座標數據包含 NaN 值（將自動處理）
- 同一 `id` 的多行代表同一影片的不同幀
- `sign_language` 欄位為分類目標標籤