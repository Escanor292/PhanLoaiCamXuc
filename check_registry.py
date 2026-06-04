from model_registry import ModelRegistry

r = ModelRegistry()
print(f'Total models: {len(r.registry["models"])}')
print(f'Best model: {r.registry["best_model"]}')
print('\nAll models:')
for m in r.registry['models']:
    print(f'  {m["model_id"]}: F1={m["metrics"]["macro_f1"]*100:.2f}%')
