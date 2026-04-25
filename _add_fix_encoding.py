"""
Auto-add 'import fix_encoding' to all affected Python files.
Run once: python _add_fix_encoding.py
"""

import os

FILES_TO_FIX = [
    'train_simple.py',
    'train_with_args.py',
    'transfer_learning.py',
    'model_phobert.py',
    'model_registry.py',
    'data_tracker.py',
    'model_sharing.py',
    'predict.py',
    'demo_phobert.py',
    'train_unified.py',
    'train.py',
    'compare_experiments.py',
    'api_server.py',
    'dataset.py',
    'register_model.py',
    'register_model_auto.py',
    'merge_data.py',
    'windows_doctor.py',
]

IMPORT_LINE = 'import fix_encoding  # Fix Windows emoji encoding'

fixed = []
already = []
skipped = []

for fname in FILES_TO_FIX:
    if not os.path.exists(fname):
        skipped.append(fname)
        continue

    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()

    if 'import fix_encoding' in content:
        already.append(fname)
        continue

    # Find best insertion point: after module docstring and comments
    lines = content.split('\n')
    insert_at = 0
    in_docstring = False
    docstring_char = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                q = stripped[:3]
                docstring_char = q
                # Check if it closes on the same line
                rest = stripped[3:]
                if q in rest:
                    insert_at = i + 1
                    continue
                in_docstring = True
                insert_at = i + 1
                continue
            elif stripped.startswith('#') or stripped == '':
                insert_at = i + 1
                continue
            else:
                break
        else:
            if docstring_char in line:
                in_docstring = False
                insert_at = i + 1
                continue

    lines.insert(insert_at, IMPORT_LINE)
    new_content = '\n'.join(lines)

    with open(fname, 'w', encoding='utf-8') as f:
        f.write(new_content)
    fixed.append(fname)

print('Fixed:')
for f in fixed:
    print(f'  [OK] {f}')
if already:
    print('Already had fix:')
    for f in already:
        print(f'  [SKIP] {f}')
if skipped:
    print('Not found:')
    for f in skipped:
        print(f'  [MISS] {f}')
print(f'\nTotal fixed: {len(fixed)} files')
