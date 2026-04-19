"""Directly patch the assembled notebook to fix Transformer issues."""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

nb_path = 'i23-2654_Assignment2_DS-A/i23-2654_Assignment2_DS-A.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']

def fix_source(src):
    # Fix 1: Remove mask from tx_model calls (avoid OOM from mask broadcasting)
    src = src.replace(
        'logits, _ = tx_model(X_cls_train.to(DEVICE), M_cls_train.to(DEVICE))',
        'logits, _ = tx_model(X_cls_train.to(DEVICE))'
    )
    src = src.replace(
        'val_logits, _ = tx_model(X_cls_val.to(DEVICE), M_cls_val.to(DEVICE))',
        'val_logits, _ = tx_model(X_cls_val.to(DEVICE))'
    )
    src = src.replace(
        'test_logits, test_attn = tx_model(X_cls_test.to(DEVICE), M_cls_test.to(DEVICE))',
        'test_logits, test_attn = tx_model(X_cls_test.to(DEVICE))'
    )
    # Fix 2: Fix any remaining unsqueeze(1).unsqueeze(2) mask issue
    src = src.replace(
        'head_mask = mask.unsqueeze(1).unsqueeze(2) if mask is not None else None',
        'head_mask = mask.unsqueeze(1) if mask is not None else None'
    )
    return src

patched = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        old_src = ''.join(cell.get('source', []))
        new_src = fix_source(old_src)
        if new_src != old_src:
            cell['source'] = [new_src]
            patched += 1
            print(f"Patched cell")
        # Clear existing outputs so cells re-execute cleanly
        cell['outputs'] = []
        cell['execution_count'] = None

print(f"\nTotal cells patched: {patched}")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook saved.")
