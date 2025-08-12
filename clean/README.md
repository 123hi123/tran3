# 手語姿態資料清理管道

此管道清理從 MediaPipe 提取的手語姿態資料，處理缺失值並為機器學習模型準備資料。

## 概述

此管道處理包含 MediaPipe 座標的 CSV 資料：
- **姿態地標點**：鼻子、左/右肩膀
- **手部地標點**：每隻手 21 個點（左/右手）含 x、y、z 座標
- **手語標籤**：對應至 `sord.csv` 中的類型

## 檔案

- `clean.py` - 主要資料清理腳本
- `gen.py` - 產生假測試資料
- `README.md` - 此說明文件

## 資料處理規則

### 1. 手部缺失值檢測（80% 閾值）
- **僅適用於 's' 類型手語**（如 `sord.csv` 中定義）
- **獨立評估**左手和右手
- **計算缺失率**使用 NaN 值（非 0.0）
- **移除整隻手部資料**如果缺失率 ≥ 80%

**範例情境：**
```
左手：85% 缺失，右手：60% 缺失 → 僅移除左手
左手：70% 缺失，右手：90% 缺失 → 僅移除右手  
左手：85% 缺失，右手：85% 缺失 → 移除雙手
左手：70% 缺失，右手：60% 缺失 → 保留雙手
```

### 2. 鼻子和肩膀插值
**針對每個影片 ID 分別處理：**
- **第一幀**：向前填充 - 找到下一個有效值
- **最後一幀**：向後填充 - 找到前一個有效值  
- **中間幀**：線性插值 - 前後有效值的平均

**範例：**
```
幀數：    0    1    2    3    4    5    6
值：     NaN  NaN  5.0  NaN  NaN  8.0  NaN
結果：    5.0  5.0  5.0  6.5  6.5  8.0  8.0
```

### 3. 手部資料政策
- **不對手部地標點進行插值**
- **保持語義準確性** - 避免產生錯誤手勢
- **尊重動作複雜性** - 手部動作過於複雜，無法安全插值

### 4. 最終缺失值處理
- **將剩餘 NaN 值填充為 0.0**
- **保留有效的 0.0 座標**（例如：鼻子在左邊緣 = x=0.0）

## 使用方法

### 基本用法
```bash
python clean.py input_data.csv
```
輸出：`output/cleaned_input_data.csv`

### 記憶體高效模式（建議）
```bash
python clean.py input_data.csv --low
```
一次處理一個影片 ID 以減少記憶體使用。

### 自訂選項
```bash
python clean.py input_data.csv \
    --low \
    --sord custom_sord.csv \
    --output custom_output.csv
```

### 命令列參數
- `input_file` - 輸入 CSV 檔案路徑（必填）
- `--sord` - 手語對應檔案路徑（預設：`sord.csv`）
- `--low` - 低記憶體模式 - 一次處理一個影片 ID
- `--output` - 自訂輸出檔案路徑（預設：`output/cleaned_<input_name>.csv`）

## 輸入資料格式

### CSV 結構
```csv
id,sign_language,nose_x,nose_y,nose_z,left_shoulder_x,left_shoulder_y,left_shoulder_z,right_shoulder_x,right_shoulder_y,right_shoulder_z,left_hand_0_x,left_hand_0_y,...
1,HELLO,0.5,0.3,0.01,0.4,0.5,,0.6,0.5,0.02,,,0.3,0.4,0.01,...
```

### 關鍵需求
- **缺失值**：空白儲存格（CSV：`,,`）代表真正的缺失資料
- **有效零值**：`0.0` 是有效座標（例如：點在影像邊緣）
- **相同 ID**：具有相同 `id` 的幀屬於同一影片，按時間排序
- **手語對應**：`sign_language` 值必須存在於 `sord.csv` 中

### sord.csv 格式
```csv
,HELLO,GOODBYE,THANK_YOU,...
person1,s,d,s,...
```
- `s` = 靜態手語（如果 80%+ 缺失則移除手部資料）
- `d` = 動態手語（無論缺失率如何都保留手部資料）

## 測試資料產生

### 產生測試資料
```bash
python gen.py --ids 5 --frames 50 --missing 0.3 --output test.csv
```

### 參數
- `--ids` - 影片 ID 數量（預設：5）
- `--frames` - 每個影片 ID 的幀數（預設：50）
- `--missing` - 缺失值比率 0.0-1.0（預設：0.3）
- `--output` - 輸出檔案名稱（預設：test_data.csv）

## 處理管道

```
輸入 CSV
    ↓
載入 sord.csv 對應
    ↓
針對每個影片 ID：
    ↓
檢查 's' 類型手語 → 計算手部缺失率 → 如果 ≥80% 則移除
    ↓
插值鼻子/肩膀座標（向前/向後/線性）
    ↓
將剩餘 NaN 填充為 0.0
    ↓
輸出至清理後的 CSV
```

## 記憶體模式

### 標準模式
- 將整個資料集載入記憶體
- 同時處理所有影片 ID
- 對較小資料集（<1GB）較快

### 低記憶體模式（`--low`）
- 一次處理一個影片 ID
- 使用分塊讀取（每塊 1000 列）
- 建議用於大型資料集或記憶體受限的情況

## 輸出

### 目錄結構
```
project/
├── input_data.csv
├── sord.csv
├── clean/
│   ├── clean.py
│   ├── gen.py
│   └── README.md
└── output/
    └── cleaned_input_data.csv
```

### 處理日誌
```
Loading data from input_data.csv
Loaded 303 sign language mappings
Processing in low memory mode (one group at a time)
Processing ID: 1, Frames: 50
Processing ID: 2, Frames: 45
Sign: HELLO - Left hand missing: 85.23%, Right hand missing: 78.90%
Removing left hand data for sign: HELLO
Processing ID: 3, Frames: 60
...
Data cleaning completed!
原始形狀：(500, 137)
清理後形狀：(500, 137)
```

## 錯誤處理

### 常見問題
- **檔案未找到**：檢查輸入檔案和 sord.csv 路徑
- **記憶體錯誤**：對大型資料集使用 `--low` 旗標
- **無效手語**：確保所有 sign_language 值都存在於 sord.csv 中
- **空白影片**：跳過沒有有效幀的影片 ID

### 驗證
- 保持原始資料結構（相同的列/行數）
- 維持每個影片 ID 內的時間順序
- 記錄所有重要操作以確保透明度

## 最佳實務

1. **在生產資料集中總是使用 `--low` 旗標**
2. **在處理前驗證 sord.csv 對應**
3. **檢查處理日誌**以了解移除的手部資料
4. **在清理前備份原始資料**
5. **首先使用 gen.py 在小型資料集上測試**

## 技術說明

### 座標系統
- **MediaPipe 格式**：x、y ∈ [0,1]、z ∈ [-1,1]
- **影像座標**：(0,0) = 左上角、(1,1) = 右下角
- **有效零值**：點可以合法地位於影像邊緣

### 效能
- **標準模式**：約 1000 幀/秒
- **低記憶體模式**：約 500 幀/秒
- **記憶體使用**：約 100MB 每 10K 幀（低記憶體模式）

### 相依套件
```bash
pip install pandas numpy
```