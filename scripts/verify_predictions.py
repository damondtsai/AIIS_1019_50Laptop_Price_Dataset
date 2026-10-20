import json

with open('data/results.json', 'r', encoding='utf-8') as f:
    res = json.load(f)

model = res['model_parameters']
intercept = model['intercept']
coefs = model['coefficients']
final_features = model['final_features']

print(f"Verifying {len(res['verification_samples'])} samples:")
max_diff = 0
for sample in res['verification_samples']:
    pred = intercept
    feat_vals = {
        'Ram_GB': sample['ram_gb'],
        'Weight_kg': sample['weight_kg'],
        'Inches': sample['inches'],
        'PPI': sample['ppi'],
        'Cpu_Speed_GHz': sample['cpu_speed_ghz'],
        'SSD_GB': sample['ssd_gb'],
        'HDD_GB': sample['hdd_gb'],
        'Touchscreen': sample['touchscreen'],
        'IPS': sample['ips']
    }
    
    comp = sample['company'] if sample['company'] in model['top_companies'] else 'Other'
    for c in model['ohe_categories']['Company_Clean']:
        feat_vals[f'Company_Clean_{c}'] = 1 if comp == c else 0
    for t in model['ohe_categories']['TypeName']:
        feat_vals[f'TypeName_{t}'] = 1 if sample['type_name'] == t else 0
    for cp in model['ohe_categories']['Cpu_Type']:
        feat_vals[f'Cpu_Type_{cp}'] = 1 if sample['cpu_type'] == cp else 0
    for g in model['ohe_categories']['Gpu_Brand']:
        feat_vals[f'Gpu_Brand_{g}'] = 1 if sample['gpu_brand'] == g else 0
    for o in model['ohe_categories']['OpSys_Clean']:
        feat_vals[f'OpSys_Clean_{o}'] = 1 if sample['opsys'] == o else 0
        
    for feat in final_features:
        pred += coefs[feat] * feat_vals.get(feat, 0)
    
    diff = abs(pred - sample['py_predicted_price'])
    max_diff = max(max_diff, diff)
    print(f"ID {sample['laptop_id']} ({sample['company']} {sample['product']}): Model={sample['py_predicted_price']}€ | Manual={pred:.2f}€ | diff={diff:.6f}")

print(f"\nMaximum difference across all verification samples: {max_diff:.8f}")
assert max_diff < 1e-2, "Difference should be negligible!"
print("Verification SUCCESSFUL!")
