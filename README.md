# Laptop Price Dataset 筆電價格預測與特徵選擇互動分析儀表板

本專案為遵循 **CRISP-DM（跨行業資料探勘標準流程）** 規範之機器學習專案，針對全球 **1,303 筆筆電硬體規格** 進行資料清洗、正則特徵工程、嚴格防洩漏的 One-Hot Encoding、三種特徵選擇方法（ANOVA F-test、RFE 遞迴特徵消除、LassoCV L1 正則化）評估、5-Fold 交叉驗證逐步特徵評估，以及無伺服器靜態架構下純前端 JavaScript 毫秒級即時價格預測器。

---

## 🌐 線上展示與成果網址

- **GitHub 儲存庫**：[https://github.com/damondtsai/AIIS_1019_50Laptop_Price_Dataset](https://github.com/damondtsai/AIIS_1019_50Laptop_Price_Dataset)
- **GitHub Pages 線上互動網站**：[https://damondtsai.github.io/AIIS_1019_50Laptop_Price_Dataset/](https://damondtsai.github.io/AIIS_1019_50Laptop_Price_Dataset/)

---

## 📊 核心成果摘要 (Key Metrics)

| 指標項目 | 數值 / 狀態 | 說明 |
| :--- | :--- | :--- |
| **資料總筆數** | **1,303 筆** | 訓練集 1,042 筆 (80%)、測試集 261 筆 (20%)，隨機種子固定 `random_state=42` |
| **原始欄位數** | **13 欄** | 排除 `laptop_ID` 與高基數 `Product` 欄位 |
| **One-Hot 編碼後總特徵數** | **38 個** | 9 個數值特徵 + 29 個 One-Hot 類別虛擬變數 |
| **5-Fold CV 最佳模型** | **k = 23** | 交叉驗證訓練集 RMSE 為 **€340.71** |
| **獨立測試集最終 RMSE** | **€355.57** | 獨立評估 261 筆測試集樣本 |
| **獨立測試集最終 MAE** | **€237.99** | 平均絕對誤差 |
| **獨立測試集決定係數 (R²)** | **0.7511** | 模型成功解釋了 75.1% 的價格變異量 |

---

## 🛠️ 資料來源與特徵工程處理規則

### 1. 資料集來源
- **原始資料集**：[MainakRepositor/Datasets laptop_price.csv](https://raw.githubusercontent.com/MainakRepositor/Datasets/master/laptop_price.csv)
- **目標變數 (Target)**：`Price_euros`（筆電終端零售價格，單位為歐元 €）

### 2. 資料清洗與數值萃取規則
- **`laptop_ID`**：純識別編號，直接剔除，避免資料外洩與虛假相關。
- **`Product` 高基數評估說明**：
  資料集中 `Product` 欄位在 1,303 筆樣本中包含高達 **618 種型號**（基數比率達 47.4%）。若將其進行 One-Hot 編碼，將引入 600+ 個極度稀疏的二元虛擬變數。在訓練樣本僅 1,042 筆的情況下，會導致嚴重的維度災難 (Curse of Dimensionality) 與過擬合。由於筆電的實質性能與成本已由 CPU、GPU、RAM、SSD/HDD、螢幕等具體規格完整體現，因此我們依據統計學習原則將 `Product` 排除於線性迴歸特徵外。
- **`Ram`**：去除字尾 "GB"，轉換為純整數 (2 ~ 64 GB)。
- **`Weight`**：去除字尾 "kg"，轉換為浮點數 (0.69 ~ 4.7 kg)。
- **`ScreenResolution`**：
  - 正則表達式擷取水平與垂直解析度（如 `1920x1080`）。
  - 結合螢幕尺寸 `Inches` 計算螢幕像素密度：$$\text{PPI} = \frac{\sqrt{W^2 + H^2}}{\text{Inches}}$$
  - 擷取 `Touchscreen`（觸控螢幕，0 或 1）與 `IPS`（廣視角面板，0 或 1）。
- **`Cpu`**：
  - 正則表達式擷取時脈頻率 `Cpu_Speed_GHz`（如 2.5 GHz）。
  - 歸納為 5 大 CPU 核心類別：`Intel Core i7`、`Intel Core i5`、`Intel Core i3`、`AMD`、`Other Intel`。
- **`Gpu`**：
  - 擷取主要核心供應商：`Intel`、`Nvidia`、`AMD`、`Other`。
- **`Memory`**：
  - 支援雙硬碟混合配置解析（如 `128GB SSD + 1TB HDD`），自動將 TB 換算為 1024 GB，分離並加總為 `SSD_GB` 與 `HDD_GB`。
- **`OpSys`**：
  - 歸納為 5 大作業系統：`Windows`、`macOS`、`Linux`、`Chrome OS`、`No OS / Other`。
- **`Company`**：
  - 嚴格僅從訓練資料中統計品牌頻次，出現次數 $\ge 10$ 之 8 家主流品牌保留（`Lenovo`, `Dell`, `HP`, `Asus`, `Acer`, `MSI`, `Toshiba`, `Apple`），其餘罕見品牌自動併入 `Other`。

### 3. 防洩漏 One-Hot Encoding 機制
- 使用 `sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False)`。
- **嚴格執行先切分、再擬合**：切分出 80% 訓練集與 20% 測試集後，所有類別頻次合併規則與 One-Hot 編碼器只在訓練集上 `fit`，測試集僅執行 `transform`。
- **編碼後特徵數**：
  - 數值特徵 (9 個)：`Ram_GB`, `Weight_kg`, `Inches`, `PPI`, `Cpu_Speed_GHz`, `SSD_GB`, `HDD_GB`, `Touchscreen`, `IPS`。
  - 類別虛擬變數 (29 個)：`Company` (9) + `TypeName` (6) + `Cpu_Type` (5) + `Gpu_Brand` (4) + `OpSys_Clean` (5)。
  - **總計特徵數 = 38 個**。

---

## 🔬 特徵選擇與逐步評估方法學

專案以 Python 實際執行並比較三種主流特徵選擇方法：

1. **ANOVA F-test (`f_regression`)**：
   - 評估單變數與價格之線性關聯性。RAM ($F = 1,280.9$) 與 SSD ($F = 850.8$) 展現出壓倒性的單變數相關度。
2. **RFE (遞迴特徵消除)**：
   - 配合 `StandardScaler` 標準化縮放與多元線性迴歸權重，逐步剔除最弱特徵。RFE 成功發掘了 `TypeName_Workstation` 與 `TypeName_Notebook` 等具有多變數互補區隔力的特徵。
3. **LassoCV (L1 正則化稀疏篩選)**：
   - 透過 5-Fold CV 尋得最佳懲罰強度 $\alpha = 4.4914$。L1 懲罰項能直接將多餘且共線的虛擬特徵權重壓縮至 0，達成自動化特徵精簡。

### 逐步特徵評估與最佳特徵數判定
- 特徵數量從 $k = 1$ 到 $k = 25$ 逐步納入。
- **特徵挑選封裝於 5-Fold CV 內部**：每次交叉驗證時，僅以該折疊的訓練分割計算特徵挑選，完全避免資訊外洩。
- **判定準則**：以 5-Fold CV 訓練集之 RMSE 作為唯一選擇標準。在 **$k = 23$** 時達到最小 CV RMSE (€340.71)；單次測試切分中 $k=24$ 雖取得 €355.42（僅差 €0.15），但依據機器學習準則，模型超參數絕不依賴測試集做挑選，以確保泛化能力。

---

## ⚡ 無伺服器純前端即時價格預測器

為相容 GitHub Pages 靜態網站架構，本專案將 Python 訓練後的最佳線性迴歸模型參數（截距、係數字典、各類別映射關係、數值上下限）完整導出為 `data/results.json` 與 `data/model_data.json`：

- **零延遲即時推論**：使用者調整品牌、類型、尺寸、螢幕規格、RAM、SSD、CPU 等選項時，前端 JavaScript 即時執行相同的特徵轉換與線性點積計算。
- **雙幣別顯示**：即時輸出歐元 (€) 與新台幣 (NT$，以匯率約 34.5 計算)。
- **主要特徵貢獻度拆解**：即時分析當前規格中「最大正向加價特徵」與「折價扣分特徵」。
- **防呆與異常規格提示**：對於極罕見或矛盾的規格組合（如非 Apple 硬體選用 macOS、低階 Celeron 搭配 64GB RAM 等），系統會自動跳出溫馨提示。
- **10 筆測試集真實樣本對照**：介面附有 10 筆真實筆電歷史規格，點擊可一鍵載入預測器，驗證 JavaScript 運算與 Python 原生預測達到小數點 2 位完全一致。

---

## 📂 專案檔案結構

```
AIIS_1019_50Laptop_Price_Dataset/
├── index.html                  # 互動式儀表板前端核心 HTML (繁體中文)
├── css/
│   └── style.css               # 深色主題、霓虹漸層與玻璃擬態 CSS 樣式表
├── js/
│   └── main.js                 # 資料加載、Chart.js 圖表渲染、即時推論運算引擎
├── data/
│   ├── laptop_price.csv        # 原始筆電規格資料集 (1,303 筆)
│   ├── results.json            # Python 建模與特徵評估完整 JSON 成果
│   └── model_data.json         # 最佳模型截距、係數矩陣與參數配置
├── scripts/
│   ├── build_results.py        # 核心分析程式：資料清理、特徵工程、CV 評估與成果導出
│   └── verify_predictions.py   # Python vs JS 數學推論一致性自動化測試腳本
├── requirements.txt            # Python 相依套件清單 (pandas, scikit-learn, numpy)
├── .gitignore                  # Git 忽略虛擬環境與快取檔
└── README.md                   # 完整專案說明文件
```

---

## 💻 本機執行與開發步驟

### 1. 複製儲存庫
```bash
git clone https://github.com/damondtsai/AIIS_1019_50Laptop_Price_Dataset.git
cd AIIS_1019_50Laptop_Price_Dataset
```

### 2. 建立並啟動 Python 虛擬環境
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安裝必要套件
```bash
pip install -r requirements.txt
```

### 4. 執行特徵工程與模型建置
```bash
python scripts/build_results.py
```
此指令會重新讀取 `data/laptop_price.csv`，執行資料前處理、5-Fold 交叉驗證逐步評估、ANOVA/RFE/LassoCV 運算，並自動更新 `data/results.json` 與 `data/model_data.json`。

### 5. 執行 Python vs JS 預測一致性測試
```bash
python scripts/verify_predictions.py
```

### 6. 啟動本機 HTTP 伺服器並預覽
```bash
python -m http.server 8000
```
開啟瀏覽器訪問：[http://localhost:8000](http://localhost:8000)

---

## 🚀 GitHub Pages 部署設定教學

本專案所有資源引用皆採用相對路徑 (`css/style.css`, `js/main.js`, `data/results.json`)，可原生相容 GitHub Pages 子路徑：

1. 登入 GitHub 並進入您的儲存庫：[damondtsai/AIIS_1019_50Laptop_Price_Dataset](https://github.com/damondtsai/AIIS_1019_50Laptop_Price_Dataset)
2. 點擊頂部選單的 **「Settings」（設定）**。
3. 在左側側邊欄找到並點選 **「Pages」**。
4. 在 **「Build and deployment」** 區塊：
   - **Source** 選擇：`Deploy from a branch`
   - **Branch** 選擇：`main` 分支，資料夾選擇 `/ (root)`
   - 點擊 **「Save」（儲存）**。
5. 稍候約 1~2 分鐘，頁面頂端將顯示線上網址：
   `https://damondtsai.github.io/AIIS_1019_50Laptop_Price_Dataset/`

---

## 📝 免責聲明 (Disclaimer)

本專案與儀表板預測器所使用之資料集為公開之歷史市場筆電規格數據（約 2017-2018 年區間），主要目的在於展示資料科學特徵工程、特徵選擇方法學之比較與互動式網頁推論架構。預測所得之價格僅供學術教學與分析研究參考，不得作為現代實際消費性電子產品交易報價之依據。
