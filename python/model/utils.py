import os
import torch
import traceback


def load_best_model(model_class, model_init_params, model_path, device):
    if not os.path.exists(model_path):
        print(f"BŁĄD KRYTYCZNY: Nie znaleziono pliku modelu w '{model_path}'")
        return None

    print(f"--- Próba załadowania modelu z pliku: {os.path.basename(model_path)} ---")

    try:
        print(f"Klasa modelu: {model_class.__name__}")
        print(f"Parametry inicjalizacyjne: {model_init_params}")
        loaded_model = model_class(**model_init_params)
    except Exception as e:
        print(
            f"\nBŁĄD KRYTYCZNY podczas inicjalizacji modelu klasą '{model_class.__name__}': {e}"
        )
        print("Sprawdź, czy parametry przekazywane do konstruktora są poprawne.")
        traceback.print_exc()
        return None

    try:
        print(f"Ładowanie wag na urządzenie: {device.type}")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)

        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        loaded_model.load_state_dict(state_dict)
        loaded_model.to(device)
        loaded_model.eval()

        print(f"Model pomyślnie załadowany i przeniesiony na urządzenie: {device}")
        return loaded_model

    except Exception as e:
        print(
            f"\nBŁĄD KRYTYCZNY podczas ładowania wag (state_dict) z '{model_path}': {e}"
        )
        print(
            "Najczęstszą przyczyną jest niezgodność architektury modelu w pamięci z tą zapisaną w pliku."
        )
        traceback.print_exc()
        return None
