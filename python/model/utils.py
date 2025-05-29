import os
import json
import torch

def load_best_model(model_class, model_init_params, model_path, device):
    if not os.path.exists(model_path):
        print(f"Błąd: Nie znaleziono pliku modelu w {model_path}")
        return None
    try:
        loaded_model = model_class(**model_init_params)
        loaded_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        loaded_model.to(device)
        loaded_model.eval()
        return loaded_model
    except Exception as e:
        print(f"Błąd podczas wczytywania lub inicjalizacji modelu z {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None