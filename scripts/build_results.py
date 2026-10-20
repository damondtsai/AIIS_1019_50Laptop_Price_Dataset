"""
Laptop Price Dataset Analysis & Results Generator
Builds all models, feature selection (ANOVA, RFE, LassoCV), CV evaluation,
and exports JSON for the static interactive dashboard.
"""

import os
import json
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import f_regression, RFE, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def clean_laptop_data(df_raw):
    df = df_raw.copy()
    
    # 1. Target and ID check
    df['Price_euros'] = df['Price_euros'].astype(float)
    
    # 2. Ram & Weight numeric
    df['Ram_GB'] = df['Ram'].astype(str).str.replace('GB', '', regex=False).str.strip().astype(int)
    df['Weight_kg'] = df['Weight'].astype(str).str.replace('kg', '', regex=False).str.strip().astype(float)
    
    # 3. Screen Resolution parsing
    def parse_screen(row):
        s = str(row['ScreenResolution'])
        touch = 1 if 'Touchscreen' in s else 0
        ips = 1 if 'IPS' in s else 0
        m = re.search(r'(\d{3,4})x(\d{3,4})', s)
        w = int(m.group(1)) if m else 1920
        h = int(m.group(2)) if m else 1080
        inches = float(row['Inches'])
        ppi = ((w**2 + h**2)**0.5) / inches if inches > 0 else 141.2
        return touch, ips, w, h, round(ppi, 2)
    
    screen_tuples = df.apply(parse_screen, axis=1)
    df['Touchscreen'] = [t[0] for t in screen_tuples]
    df['IPS'] = [t[1] for t in screen_tuples]
    df['Res_Width'] = [t[2] for t in screen_tuples]
    df['Res_Height'] = [t[3] for t in screen_tuples]
    df['PPI'] = [t[4] for t in screen_tuples]
    
    # 4. CPU parsing
    def parse_cpu(s):
        s = str(s)
        sm = re.search(r'(\d+(?:\.\d+)?)GHz', s)
        speed = float(sm.group(1)) if sm else 2.3
        if 'Intel Core i7' in s:
            brand = 'Intel Core i7'
        elif 'Intel Core i5' in s:
            brand = 'Intel Core i5'
        elif 'Intel Core i3' in s:
            brand = 'Intel Core i3'
        elif 'AMD' in s:
            brand = 'AMD'
        else:
            brand = 'Other Intel'
        return brand, round(speed, 2)
    
    cpu_tuples = df['Cpu'].apply(parse_cpu)
    df['Cpu_Type'] = [t[0] for t in cpu_tuples]
    df['Cpu_Speed_GHz'] = [t[1] for t in cpu_tuples]
    
    # 5. GPU parsing
    def parse_gpu(s):
        s = str(s).strip()
        first = s.split()[0]
        if first in ['Intel', 'Nvidia', 'AMD']:
            return first
        return 'Other'
    df['Gpu_Brand'] = df['Gpu'].apply(parse_gpu)
    
    # 6. Memory parsing
    def parse_memory(s):
        ssd, hdd = 0, 0
        parts = str(s).split('+')
        for part in parts:
            part = part.strip()
            m = re.search(r'(\d+(?:\.\d+)?)\s*(GB|TB)', part)
            if m:
                val = float(m.group(1))
                unit = m.group(2)
                gb = int(val * 1024) if unit == 'TB' else int(val)
                if 'SSD' in part:
                    ssd += gb
                elif 'HDD' in part:
                    hdd += gb
        return ssd, hdd
    
    mem_tuples = df['Memory'].apply(parse_memory)
    df['SSD_GB'] = [t[0] for t in mem_tuples]
    df['HDD_GB'] = [t[1] for t in mem_tuples]
    
    # 7. OpSys parsing
    def parse_opsys(s):
        s = str(s).strip()
        if 'Windows' in s:
            return 'Windows'
        elif 'Mac' in s or 'macOS' in s:
            return 'macOS'
        elif 'Linux' in s:
            return 'Linux'
        elif 'Chrome' in s:
            return 'Chrome OS'
        else:
            return 'No OS / Other'
    df['OpSys_Clean'] = df['OpSys'].apply(parse_opsys)
    
    return df

def main():
    csv_path = 'data/laptop_price.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    
    # Read raw data
    try:
        df_raw = pd.read_csv(csv_path, encoding='latin-1')
    except Exception:
        df_raw = pd.read_csv(csv_path, encoding='utf-8')
    
    total_samples = len(df_raw)
    raw_columns = df_raw.columns.tolist()
    
    # Clean and feature engineering
    df_cleaned = clean_laptop_data(df_raw)
    
    # Split train/test (80/20, random_state=42) strictly first
    train_df, test_df = train_test_split(df_cleaned, test_size=0.2, random_state=42)
    
    # Determine rare categories purely on training data!
    company_counts = train_df['Company'].value_counts()
    top_companies = company_counts[company_counts >= 10].index.tolist()
    
    def group_company(subset):
        d = subset.copy()
        d['Company_Clean'] = d['Company'].apply(lambda c: c if c in top_companies else 'Other')
        return d
    
    train_df = group_company(train_df)
    test_df = group_company(test_df)
    
    num_features = ['Ram_GB', 'Weight_kg', 'Inches', 'PPI', 'Cpu_Speed_GHz', 'SSD_GB', 'HDD_GB', 'Touchscreen', 'IPS']
    cat_features = ['Company_Clean', 'TypeName', 'Cpu_Type', 'Gpu_Brand', 'OpSys_Clean']
    
    # Fit OneHotEncoder strictly on train
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(train_df[cat_features])
    encoded_cat_names = ohe.get_feature_names_out(cat_features).tolist()
    all_feature_names = num_features + encoded_cat_names
    
    # Matrix construction
    X_train_num = train_df[num_features].values
    X_train_cat = ohe.transform(train_df[cat_features])
    X_train = np.hstack([X_train_num, X_train_cat])
    y_train = train_df['Price_euros'].values
    
    X_test_num = test_df[num_features].values
    X_test_cat = ohe.transform(test_df[cat_features])
    X_test = np.hstack([X_test_num, X_test_cat])
    y_test = test_df['Price_euros'].values
    
    # 1. Full model baseline
    lr_full = LinearRegression()
    lr_full.fit(X_train, y_train)
    y_pred_full = lr_full.predict(X_test)
    full_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_full))
    full_test_r2 = r2_score(y_test, y_pred_full)
    full_test_mae = mean_absolute_error(y_test, y_pred_full)
    
    # 2. ANOVA F-test
    f_scores, p_values = f_regression(X_train, y_train)
    anova_ranking_idx = np.argsort(-f_scores)
    anova_list = []
    for rank, idx in enumerate(anova_ranking_idx):
        anova_list.append({
            'rank': rank + 1,
            'feature': all_feature_names[idx],
            'f_score': round(float(f_scores[idx]), 2),
            'p_value': float(f"{p_values[idx]:.2e}"),
            'description': f"F 值 {f_scores[idx]:.1f}，顯示該特徵與價格具有高度單變數線性相關性。"
        })
    
    # 3. RFE (with StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rfe_est = LinearRegression()
    rfe = RFE(estimator=rfe_est, n_features_to_select=1)
    rfe.fit(X_train_scaled, y_train)
    rfe_ranking_idx = np.argsort(rfe.ranking_)
    rfe_list = []
    for rank, idx in enumerate(rfe_ranking_idx):
        rfe_list.append({
            'rank': rank + 1,
            'feature': all_feature_names[idx],
            'rfe_rank': int(rfe.ranking_[idx]),
            'description': f"RFE 依據多元迴歸權重遞迴消除後第 {rfe.ranking_[idx]} 名保留之特徵。"
        })
    
    # 4. LassoCV (with StandardScaler)
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=5000)
    lasso_cv.fit(X_train_scaled, y_train)
    lasso_coefs = lasso_cv.coef_
    lasso_ranking_idx = np.argsort(-np.abs(lasso_coefs))
    lasso_list = []
    for rank, idx in enumerate(lasso_ranking_idx):
        coef_val = float(lasso_coefs[idx])
        lasso_list.append({
            'rank': rank + 1,
            'feature': all_feature_names[idx],
            'coef_standardized': round(coef_val, 2),
            'abs_coef': round(abs(coef_val), 2),
            'is_selected': bool(abs(coef_val) > 1e-4),
            'description': f"L1 正則化標準化係數 {coef_val:+.2f}，{'納入模型' if abs(coef_val) > 1e-4 else '係數被收縮為 0 剔除'}。"
        })
    
    # 5. Stepwise Feature Evaluation (k = 1 to 25 or up to len(all_feature_names))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    stepwise_records = []
    max_k = min(25, len(all_feature_names))
    
    for k in range(1, max_k + 1):
        pipe = Pipeline([
            ('select', SelectKBest(f_regression, k=k)),
            ('lr', LinearRegression())
        ])
        
        # 5-Fold CV on training split
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
        cv_rmse = float(-np.mean(cv_scores))
        
        # Train on full X_train and evaluate on hold-out X_test
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        test_r2 = float(r2_score(y_test, y_pred))
        test_mae = float(mean_absolute_error(y_test, y_pred))
        
        selector = pipe.named_steps['select']
        selected_mask = selector.get_support()
        selected_names = [all_feature_names[i] for i, m in enumerate(selected_mask) if m]
        
        stepwise_records.append({
            'k': k,
            'features': selected_names,
            'features_display': ', '.join(selected_names[-3:]) + (f" (共 {k} 個)" if k > 3 else ""),
            'cv_rmse': round(cv_rmse, 2),
            'test_rmse': round(test_rmse, 2),
            'test_r2': round(test_r2, 4),
            'test_mae': round(test_mae, 2)
        })
    
    # Identify CV-selected best model vs test-lowest model
    best_cv_idx = int(np.argmin([rec['cv_rmse'] for rec in stepwise_records]))
    best_cv_rec = stepwise_records[best_cv_idx]
    
    best_test_idx = int(np.argmin([rec['test_rmse'] for rec in stepwise_records]))
    best_test_rec = stepwise_records[best_test_idx]
    
    # Fit the CV-selected model specifically for export & real-time client predictor
    best_k = best_cv_rec['k']
    final_pipe = Pipeline([
        ('select', SelectKBest(f_regression, k=best_k)),
        ('lr', LinearRegression())
    ])
    final_pipe.fit(X_train, y_train)
    final_selector = final_pipe.named_steps['select']
    final_mask = final_selector.get_support()
    final_features = [all_feature_names[i] for i, m in enumerate(final_mask) if m]
    
    final_lr = final_pipe.named_steps['lr']
    intercept = float(final_lr.intercept_)
    coefficients = [float(c) for c in final_lr.coef_]
    
    # Test predictions and residuals on test set for visualization
    y_test_pred = final_pipe.predict(X_test)
    residuals = y_test - y_test_pred
    
    scatter_data = []
    for i in range(len(y_test)):
        row_orig = test_df.iloc[i]
        scatter_data.append({
            'laptop_id': int(row_orig['laptop_ID']),
            'company': str(row_orig['Company']),
            'product': str(row_orig['Product']),
            'type_name': str(row_orig['TypeName']),
            'actual': round(float(y_test[i]), 2),
            'predicted': round(float(y_test_pred[i]), 2),
            'residual': round(float(residuals[i]), 2),
            'ram': int(row_orig['Ram_GB']),
            'weight': float(row_orig['Weight_kg']),
            'screen': str(row_orig['ScreenResolution']),
            'cpu': str(row_orig['Cpu']),
            'gpu': str(row_orig['Gpu'])
        })
    
    # Verification samples (10 test samples)
    test_verification_samples = []
    # Pick 10 representative indices across price ranges
    sample_indices = np.linspace(0, len(test_df) - 1, 10, dtype=int)
    for idx in sample_indices:
        row = test_df.iloc[idx]
        actual_p = float(row['Price_euros'])
        py_pred = float(y_test_pred[idx])
        test_verification_samples.append({
            'laptop_id': int(row['laptop_ID']),
            'company': str(row['Company']),
            'product': str(row['Product']),
            'type_name': str(row['TypeName']),
            'inches': float(row['Inches']),
            'ram_gb': int(row['Ram_GB']),
            'weight_kg': float(row['Weight_kg']),
            'touchscreen': int(row['Touchscreen']),
            'ips': int(row['IPS']),
            'ppi': float(row['PPI']),
            'cpu_type': str(row['Cpu_Type']),
            'cpu_speed_ghz': float(row['Cpu_Speed_GHz']),
            'gpu_brand': str(row['Gpu_Brand']),
            'ssd_gb': int(row['SSD_GB']),
            'hdd_gb': int(row['HDD_GB']),
            'opsys': str(row['OpSys_Clean']),
            'actual_price': round(actual_p, 2),
            'py_predicted_price': round(py_pred, 2)
        })
    
    # Feature Engineering Mapping documentation
    feature_mapping = [
        {
            'raw_column': 'laptop_ID',
            'role': '識別碼 (Identifier)',
            'treatment': '直接剔除，避免資料洩漏或虛假相關',
            'encoded_count': 0,
            'features': []
        },
        {
            'raw_column': 'Product',
            'role': '型號 (High-cardinality)',
            'treatment': '基數高達 618 類（超過 47%），若 One-Hot 編碼將造成維度爆炸與嚴重過擬合，故不納入線性模型，並於系統說明',
            'encoded_count': 0,
            'features': []
        },
        {
            'raw_column': 'Ram',
            'role': '數值特徵',
            'treatment': '擷取數值單位 GB，轉換為整數 (2 ~ 64 GB)',
            'encoded_count': 1,
            'features': ['Ram_GB']
        },
        {
            'raw_column': 'Weight',
            'role': '數值特徵',
            'treatment': '移除 "kg" 後綴，轉換為浮點數 (0.69 ~ 4.7 kg)',
            'encoded_count': 1,
            'features': ['Weight_kg']
        },
        {
            'raw_column': 'Inches',
            'role': '數值特徵',
            'treatment': '螢幕對角線吋數，保留原始數值並用於計算 PPI',
            'encoded_count': 1,
            'features': ['Inches']
        },
        {
            'raw_column': 'ScreenResolution',
            'role': '複合文字特徵',
            'treatment': '正規表示式擷取寬高解析度，計算像素密度 PPI = √(W²+H²)/Inches，並擷取 Touchscreen 與 IPS 標籤',
            'encoded_count': 3,
            'features': ['PPI', 'Touchscreen', 'IPS']
        },
        {
            'raw_column': 'Cpu',
            'role': '複合文字特徵',
            'treatment': '擷取處理器時脈 (GHz) 作為數值特徵；歸納為 5 大 CPU 類別 (Intel i7/i5/i3, AMD, Other Intel) 後進行 One-Hot 編碼',
            'encoded_count': 6,
            'features': ['Cpu_Speed_GHz', 'Cpu_Type (5 欄位)']
        },
        {
            'raw_column': 'Gpu',
            'role': '文字特徵',
            'treatment': '擷取 GPU 核心供應商 (Intel, Nvidia, AMD, Other) 進行 One-Hot 編碼',
            'encoded_count': 4,
            'features': ['Gpu_Brand (4 欄位)']
        },
        {
            'raw_column': 'Memory',
            'role': '複合儲存特徵',
            'treatment': '解析分割字串，分離並合計 SSD 容量 (GB) 與傳統 HDD 容量 (GB)',
            'encoded_count': 2,
            'features': ['SSD_GB', 'HDD_GB']
        },
        {
            'raw_column': 'OpSys',
            'role': '類別特徵',
            'treatment': '整併 Windows, macOS, Linux, Chrome OS, No OS / Other 後進行 One-Hot 編碼',
            'encoded_count': 5,
            'features': ['OpSys_Clean (5 欄位)']
        },
        {
            'raw_column': 'Company',
            'role': '類別特徵',
            'treatment': '於訓練集統計，出現次數 ≥ 10 之主流品牌 (8 家) 保留，罕見品牌合併為 Other，進行 One-Hot 編碼',
            'encoded_count': 9,
            'features': ['Company_Clean (9 欄位)']
        },
        {
            'raw_column': 'TypeName',
            'role': '類別特徵',
            'treatment': '筆電類型 (Notebook, Gaming, Ultrabook, 2 in 1, Workstation, Netbook) 進行 One-Hot 編碼',
            'encoded_count': 6,
            'features': ['TypeName (6 欄位)']
        }
    ]
    
    # Model Export Data for Frontend Realtime Predictor
    frontend_model_data = {
        'model_name': 'Scikit-Learn LinearRegression (CV Selected)',
        'best_k': best_k,
        'intercept': intercept,
        'coefficients': dict(zip(final_features, coefficients)),
        'final_features': final_features,
        'all_feature_names': all_feature_names,
        'top_companies': top_companies,
        'ohe_categories': {
            'Company_Clean': top_companies + ['Other'],
            'TypeName': list(train_df['TypeName'].unique()),
            'Cpu_Type': ['Intel Core i7', 'Intel Core i5', 'Intel Core i3', 'AMD', 'Other Intel'],
            'Gpu_Brand': ['Intel', 'Nvidia', 'AMD', 'Other'],
            'OpSys_Clean': ['Windows', 'macOS', 'Linux', 'Chrome OS', 'No OS / Other']
        },
        'num_feature_stats': {
            'Ram_GB': {'min': 2, 'max': 64, 'default': 8, 'step': 2},
            'Weight_kg': {'min': 0.6, 'max': 5.0, 'default': 2.0, 'step': 0.05},
            'Inches': {'min': 10.1, 'max': 18.4, 'default': 15.6, 'step': 0.1},
            'PPI': {'min': 90.0, 'max': 360.0, 'default': 141.2, 'step': 0.1},
            'Cpu_Speed_GHz': {'min': 0.9, 'max': 4.0, 'default': 2.5, 'step': 0.1},
            'SSD_GB': {'min': 0, 'max': 2048, 'default': 256, 'step': 128},
            'HDD_GB': {'min': 0, 'max': 2048, 'default': 0, 'step': 500},
            'Touchscreen': {'min': 0, 'max': 1, 'default': 0},
            'IPS': {'min': 0, 'max': 1, 'default': 0}
        }
    }
    
    # Complete results payload
    dashboard_payload = {
        'meta': {
            'dataset_name': 'Laptop Price Dataset (1303 筆)',
            'source_url': 'https://raw.githubusercontent.com/MainakRepositor/Datasets/master/laptop_price.csv',
            'target_variable': 'Price_euros (€)',
            'random_state': 42,
            'test_size': 0.2,
            'currency': '€'
        },
        'summary_metrics': {
            'total_samples': total_samples,
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'raw_columns_count': len(raw_columns),
            'encoded_features_count': len(all_feature_names),
            'best_cv_k': best_k,
            'best_cv_rmse': best_cv_rec['cv_rmse'],
            'best_cv_test_rmse': best_cv_rec['test_rmse'],
            'best_cv_test_r2': best_cv_rec['test_r2'],
            'best_cv_test_mae': best_cv_rec['test_mae'],
            'best_test_k': best_test_rec['k'],
            'best_test_rmse': best_test_rec['test_rmse'],
            'best_test_r2': best_test_rec['test_r2'],
            'full_model_test_rmse': round(full_test_rmse, 2),
            'full_model_test_r2': round(full_test_r2, 4),
            'full_model_test_mae': round(full_test_mae, 2)
        },
        'stepwise_evaluation': stepwise_records,
        'feature_selection_methods': {
            'anova': anova_list,
            'rfe': rfe_list,
            'lasso': {
                'alpha': round(float(lasso_cv.alpha_), 4),
                'features': lasso_list
            }
        },
        'feature_mapping': feature_mapping,
        'scatter_data': scatter_data,
        'verification_samples': test_verification_samples,
        'model_parameters': frontend_model_data
    }
    
    # Save outputs
    os.makedirs('data', exist_ok=True)
    with open('data/results.json', 'w', encoding='utf-8') as f:
        json.dump(dashboard_payload, f, ensure_ascii=False, indent=2)
    
    with open('data/model_data.json', 'w', encoding='utf-8') as f:
        json.dump(frontend_model_data, f, ensure_ascii=False, indent=2)
        
    print(f"Results successfully saved to data/results.json and data/model_data.json!")
    print(f"Total samples: {total_samples}")
    print(f"Encoded features: {len(all_feature_names)}")
    print(f"Best CV k={best_k}: CV RMSE = €{best_cv_rec['cv_rmse']}, Test RMSE = €{best_cv_rec['test_rmse']}, Test R2 = {best_cv_rec['test_r2']}")
    print(f"Single test best k={best_test_rec['k']}: Test RMSE = €{best_test_rec['test_rmse']}, Test R2 = {best_test_rec['test_r2']}")

if __name__ == '__main__':
    main()
