# ST-GCN 手語辨識專案

## 系統需求

### 硬體規格
本專案已針對以下硬體配置進行優化：
- **CPU**: Intel(R) Core(TM) i7-9700 (8核心, 3.00GHz)
- **RAM**: 48 GB DDR4-2666
- **GPU**: NVIDIA RTX A2000 (6 GB GDDR6)

### 軟體需求
- Python 3.11 或 3.12 (推薦)
- CUDA 11.8 支援
- Conda 環境管理器

## 環境需求

### 前置條件
- 已建立 `sign_language` conda 環境
- 已安裝 PyTorch (CUDA 11.8)、numpy、pandas、scikit-learn、matplotlib、seaborn、opencv-python、mediapipe

### 安裝專案需要的額外套件
```bash
# 啟動環境
conda activate sign_language

# 安裝進度條工具
pip install tqdm

# 可選：開發工具
pip install jupyter notebook ipykernel tensorboard h5py
```

## 專案結構
```
st_gcn_project/
├── data/
│   ├── raw/           # 放置 train.csv 和 val.csv 原始資料
│   └── processed/     # 處理後的資料將儲存在這裡
├── models/            # 訓練好的模型將儲存在這裡
├── src/               # 原始程式碼
│   ├── data_processor.py    # 資料處理器
│   ├── stgcn_model.py      # ST-GCN 模型
│   ├── train.py            # 訓練腳本
│   └── evaluate.py         # 評估腳本
├── results/           # 評估結果
├── requirements.txt   # Python 套件需求
├── run_pipeline.py    # 完整流程執行腳本
└── test_setup.py      # 設定測試腳本
```

## 硬體優化配置

### GPU 記憶體管理 (RTX A2000 6GB)
本專案已針對 6GB 顯存進行優化：
- **批次大小**: 16 (適合 6GB 顯存)
- **梯度累積**: 啟用，以達到等效更大批次大小
- **混合精度訓練**: 使用 AMP 減少記憶體使用
- **記憶體清理**: 自動清理未使用的快取

### CPU 多執行緒優化
- **DataLoader 工作執行緒**: 8 (對應 8 核心 CPU)
- **PyTorch 執行緒**: 優化為 8 執行緒
- **數據預處理**: 多程序並行處理

## 資料格式要求

### 輸入 CSV 檔案
請將 CSV 檔案放在 `data/raw/` 資料夾中：
- `train.csv` - 訓練資料
- `val.csv` - 驗證資料

### CSV 欄位格式
每個 CSV 檔案應包含以下欄位：

**必需欄位：**
- **id**：影片/序列的唯一識別符
- **sign_language**：目標手語分類標籤（字串或整數）
- **骨架座標**：手部骨架關節的 3D 座標

**實際欄位格式：**
```
id,sign_language,nose_x,nose_y,nose_z,left_shoulder_x,left_shoulder_y,left_shoulder_z,
right_shoulder_x,right_shoulder_y,right_shoulder_z,left_hand_0_x,left_hand_0_y,left_hand_0_z,
left_hand_1_x,left_hand_1_y,left_hand_1_z,...,left_hand_20_x,left_hand_20_y,left_hand_20_z,
right_hand_0_x,right_hand_0_y,right_hand_0_z,...,right_hand_20_x,right_hand_20_y,right_hand_20_z
```

**資料屬性：**
- **每一行代表一幀**：同一個 `id` 的多行代表一個影片的時序幀
- **手部關節**：每隻手有 21 個關節點 (hand_0 到 hand_20)
- **身體關節**：包含鼻子、左右肩膀作為參考點
- **允許缺失值**：NaN 值將透過遮罩處理
- **額外欄位**：如有其他欄位會自動忽略
- **座標單位**：應使用一致的單位（像素或正規化座標）

## 資料處理流程

本專案實現了以下完整的資料處理管道：

1. **影片分組**：根據 `id` 欄位將幀分組為完整影片序列
2. **NaN 處理**：將缺失值轉換為二進位遮罩
3. **座標遮罩**：將座標乘以遮罩以處理有效關節
4. **正規化與中心化**：標準化座標數據
5. **時間對齊**：將每個影片序列對齊到固定長度 T（padding 或截斷）
6. **輸出格式**：X[C,T,V,M] 其中：
   - C：通道數（3 代表 x,y,z）
   - T：時間幀數（固定為 20）
   - V：頂點/關節數（鼻子 + 肩膀 + 雙手 = 45 關節）
   - M：最大人數（設為 1 表示單人）

## 訓練規則

### 時序對齊規則 (T=20)

本專案採用嚴格的時序對齊規則，確保所有影片序列統一為 **20 幀**：

#### 1. **不足 20 幀**
- **處理方式**：在序列尾端補零至 20 幀
- **補零內容**：補充的幀為全零向量（所有座標為 0）
- **遮罩處理**：如有 mask，補零幀的 mask 值設為 0

#### 2. **剛好 20 幀** 
- **處理方式**：原樣使用，不做任何修改

#### 3. **超過 20 幀**
採用智能視窗選取策略：

**基本規則：**
- 先取中間的 20 幀（左右平均丟棄）

**視窗調整規則：**
- 若左側被丟 >15 幀 → 視窗向左移 5 幀
- 若右側被丟 >15 幀 → 視窗向右移 5 幀
- 視窗起點限制在 [0, 總幀數-20] 範圍內

**範例：**
```
總幀數 50，中間視窗 [15:35]
- 左側丟棄：15 幀 (≤15) → 無調整
- 右側丟棄：15 幀 (≤15) → 無調整
- 最終視窗：[15:35]

總幀數 60，中間視窗 [20:40] 
- 左側丟棄：20 幀 (>15) → 向左移 5 幀
- 調整後視窗：[15:35]

總幀數 100，中間視窗 [40:60]
- 左側丟棄：40 幀 (>15) → 向左移 5 幀  
- 調整後視窗：[35:55]
```

### 資料增強規則

- **關節丟棄（Joint Dropout）**：訓練時隨機丟棄關節，增強模型泛化能力
- **時間丟棄（Temporal Dropout）**：訓練時隨機遮蔽時間幀
- **座標正規化**：統一座標尺度，提升訓練穩定性

## ST-GCN 模型特色

實現的 ST-GCN 模型包含以下進階功能：

- **閘控機制（Gating）**：透過遮罩乘法控制資訊流
- **加權池化（Weighted Pooling）**：智能的全域平均池化
- **群組正規化（GroupNorm）**：改善訓練穩定性
- **關節丟棄（Joint Dropout）**：訓練時隨機丟棄關節增強泛化
- **時間丟棄（Temporal Dropout）**：訓練時隨機丟棄時間幀
- **混合精度訓練**：自動減少記憶體使用並加速訓練

## 使用方式

### 快速開始
```bash
# 1. 啟動環境
conda activate sign_language

# 2. 進入專案目錄
cd st_gcn_project

# 3. 測試設定（推薦）
python test_setup.py

# 4. 執行完整流程
python run_pipeline.py  # 使用默認 v1 版本
# 或
python run_pipeline.py --version=v2  # 使用修復版本 v2
```

### 版本選擇

本專案提供兩個版本的 pipeline：

#### v1 (默認版本)
```bash
python run_pipeline.py
```
- 原始實作版本
- 適用於一般使用

#### v2 (修復版本) - **推薦**
```bash
python run_pipeline.py --version=v2
```
- **修復的問題**：
  - ✅ 梯度累積邏輯錯誤（防止梯度殘留）
  - ✅ 0值處理策略（只將 NaN 視為缺失，保留真實 0 值）
  - ✅ 統計量計算錯誤（正確處理全部缺失的維度）
  - ✅ 記憶體管理優化（減少頻繁的記憶體清理）
  - ✅ 改善錯誤處理（torch.compile 失敗回退）
  - ✅ 新增數據驗證功能
- **推薦使用**：包含所有已知問題的修復

### 分步執行

#### 使用 v1 版本
```bash
# 啟動環境
conda activate sign_language

# 1. 資料處理
python src/data_processor.py

# 2. 模型訓練
python src/train.py

# 3. 模型評估
python src/evaluate.py
```

#### 使用 v2 版本（推薦）
```bash
# 啟動環境
conda activate sign_language

# 1. 資料處理
python src/data_processor_v2.py

# 2. 模型訓練
python src/train_v2.py

# 3. 模型評估
python src/evaluate.py
```

## 訓練設定 (已針對硬體優化)

### RTX A2000 優化設定
- **批次大小**: 16 (適合 6GB 顯存)
- **梯度累積步數**: 2 (等效批次大小 32)
- **學習率**: 0.001
- **權重衰減**: 1e-4
- **最大訓練週期**: 100
- **早停耐心值**: 15
- **混合精度**: 啟用 (AMP)
- **記憶體優化**: 啟用

### CPU 優化設定
- **DataLoader 執行緒**: 8 (對應 8 核心)
- **數據預處理**: 多程序並行

## 效能監控

### GPU 記憶體監控
```bash
# 監控 GPU 使用情況
nvidia-smi -l 1

# 或在 Python 中監控
python -c "import torch; print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB')"
```

### 系統資源監控
```bash
# CPU 和記憶體監控
htop

# 或
top
```

## 評估指標

評估腳本將計算並輸出：

### 基本指標
- **準確率（Accuracy）**：整體分類正確率
- **宏觀 F1 分數（Macro F1-Score）**：各類別 F1 分數的平均

### 混淆矩陣分析
- **每個類別最混淆的類別及次數**
- **相互混淆對列表**：彼此是彼此第一混淆的類別對

### 輸出檔案
- `results/evaluation_results.pkl`：完整評估結果
- `results/confusion_matrix.png`：混淆矩陣視覺化
- `results/evaluation_report.txt`：詳細文字報告
- `results/training_history.png`：訓練歷史圖表

## 疑難排解

### 常見問題

#### GPU 記憶體不足
```bash
# 如果遇到 CUDA out of memory 錯誤
# 1. 減少批次大小 (在 src/train.py 中修改)
batch_size = 8  # 從 16 減少到 8

# 2. 啟用梯度檢查點 (已在模型中實現)
```

#### CPU 效能調優
```bash
# 設定 PyTorch 執行緒數
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

#### 記憶體不足
```bash
# 如果系統記憶體不足，減少 DataLoader 執行緒數
num_workers = 4  # 從 8 減少到 4
```

## 進階設定

### 自訂硬體配置
如果您的硬體配置不同，請修改以下檔案：

#### `src/train.py` - 訓練參數
```python
# 根據您的 GPU 記憶體調整
batch_size = 16  # RTX A2000 6GB 適用
gradient_accumulation_steps = 2

# 根據您的 CPU 核心數調整
num_workers = 8  # 對應 8 核心 CPU
```

#### `src/data_processor.py` - 資料處理
```python
# 多程序並行處理
n_processes = 8  # 對應 CPU 核心數
```

## 效能預期

基於您的硬體配置：
- **訓練速度**: 約 2-3 秒/批次 (RTX A2000)
- **記憶體使用**: 約 4-5 GB GPU 記憶體
- **CPU 使用**: 充分利用 8 核心進行資料處理
- **總訓練時間**: 視資料大小而定，通常 1-3 小時

## 開發環境

### Jupyter 使用
```bash
# 如果要使用 Jupyter 開發
conda activate sign_language
pip install jupyter
jupyter notebook
```

### 模型視覺化
```bash
# 使用 TensorBoard 查看訓練進度
pip install tensorboard
tensorboard --logdir=models/tensorboard_logs
```

## 注意事項

- 所有路徑皆使用相對路徑，避免路徑錯誤
- 自動偵測並使用 GPU (RTX A2000)
- 自動保存最佳模型與訓練歷史
- 詳細的錯誤處理與進度顯示
- 完整的評估報告與視覺化
- 針對 6GB 顯存進行記憶體優化
- 充分利用 8 核心 CPU 並行處理