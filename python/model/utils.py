import os
import torch
import traceback


def load_best_model(model_class, model_init_params, model_path, device):
    if not os.path.exists(model_path):
        print(f"Błąd: Nie znaleziono pliku modelu w {model_path}")
        return None
    try:
        print(f"Ładowanie modelu klasy: {model_class}")
        print(f"Parametry inicjalizacyjne: {model_init_params}")
        loaded_model = model_class(**model_init_params)

        if device.type == "cuda":
            state_dict = torch.load(
                model_path,
                map_location=lambda storage, loc: storage.cuda(device.index),
                weights_only=True,
            )
        else:
            state_dict = torch.load(
                model_path, map_location=torch.device("cpu"), weights_only=True
            )

        loaded_model.load_state_dict(state_dict)
        loaded_model.to(
            device
        )
        loaded_model.eval()
        print(f"Model pomyślnie załadowany i przeniesiony na urządzenie: {device}")
        return loaded_model
    except Exception as e:
        print(f"Błąd podczas wczytywania lub inicjalizacji modelu z {model_path}: {e}")
        traceback.print_exc()
        return None
