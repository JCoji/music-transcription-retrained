import torch


def batch_validate(loader):
    """
    Funkcja do walidacji i wyświetlania informacji o pierwszym batchu danych z loadera.

    Iteruje przez pierwszy batch pobrany z obiektu `loader`, wyświetlając
    szczegółowe informacje o każdym tensorze w batchu, takie jak typ danych,
    urządzenie, kształt, wartości minimalne/maksymalne/średnie oraz
    specyficzne dane dla wybranych kluczy.

    Args:
        loader: Obiekt typu DataLoader (lub podobny), z którego można pobrać batch danych.
    """
    print(f"Pobieranie pierwszego batcha z {type(loader).__name__}...")
    batch = next(iter(loader))

    print(f"Klucze batcha: {list(batch.keys())}")
    first_tensor_key = next(iter(batch))
    print(
        f"Rozmiar batcha (na podstawie klucza '{first_tensor_key}'): {len(batch[first_tensor_key])}"
    )

    for key, tensor in batch.items():
        print(f"\n--- Klucz: '{key}' ---")

        if hasattr(tensor, "dtype"):
            print(f"  Typ danych (dtype): {tensor.dtype}")
            print(f"  Urządzenie (device): {tensor.device}")
            print(f"  Kształt tensora: {tensor.shape}")

            min_val = tensor.min().item()
            max_val = tensor.max().item()
            min_str = f"{min_val:.4f}" if isinstance(min_val, float) else str(min_val)
            max_str = f"{max_val:.4f}" if isinstance(max_val, float) else str(max_val)
            print(f"  Wartość minimalna: {min_str}")
            print(f"  Wartość maksymalna: {max_str}")

            if tensor.dtype.is_floating_point:
                mean_val = tensor.mean().item()
                print(f"  Wartość średnia: {mean_val:.4f}")

            if key == "feature_lengths":
                print(f"  Długości sekwencji: {tensor.tolist()}")
                print(
                    f"  Min/Max długość: {tensor.min().item()} / {tensor.max().item()}"
                )

            elif key == "mask":
                print(f"  Unikalne wartości maski: {torch.unique(tensor).tolist()}")
                # Sprawdzenie spójności maski z długością cech dla pierwszej próbki
                if (
                    "feature_lengths" in batch
                    and len(batch["feature_lengths"]) > 0
                    and tensor.ndim > 0
                    and tensor.shape[0] > 0
                ):
                    expected_len = batch["feature_lengths"][0].item()
                    mask_sum = tensor[0].sum().item()
                    print(
                        f"  Suma maski dla próbki 0: {mask_sum} (oczekiwana: {expected_len})"
                    )

            elif key in ["notes", "onsets"]:
                non_zero_count = torch.sum(tensor != 0).item()
                total_count = tensor.numel()
                # Obliczenie "rzadkości" tensora (procent zer)
                sparsity = (
                    (1 - non_zero_count / total_count) * 100
                    if total_count > 0
                    else 100.0
                )
                print(
                    f"  Liczba elementów niezerowych: {non_zero_count} / {total_count}"
                )
                print(f"  'Rzadkość' tensora (procent zer): {sparsity:.2f}%")

                if tensor.ndim >= 3:
                    print(
                        f"  Fragment [0, :5, :10]:\n{tensor[0, :5, :10].cpu().numpy()}"
                    )
                elif tensor.ndim == 2:
                    print(f"  Fragment [:5, :10]:\n{tensor[:5, :10].cpu().numpy()}")
                elif tensor.ndim == 1:
                    print(f"  Fragment [:10]:\n{tensor[:10].cpu().numpy()}")
                else:  # tensor.ndim == 0 (skalar) lub inny przypadek
                    print(
                        f"  Wartość skalarna lub nietypowy wymiar: {tensor.cpu().numpy()}"
                    )

            elif key == "contours":
                if tensor.ndim >= 3:
                    print(f"  Fragment [0, :5, :5]:\n{tensor[0, :5, :5].cpu().numpy()}")
                elif tensor.ndim == 2:
                    print(f"  Fragment [:5, :5]:\n{tensor[:5, :5].cpu().numpy()}")
                elif tensor.ndim == 1:
                    print(f"  Fragment [:5]:\n{tensor[:5].cpu().numpy()}")
                else:
                    print(
                        f"  Wartość skalarna lub nietypowy wymiar: {tensor.cpu().numpy()}"
                    )

            elif key == "features":
                if tensor.ndim >= 3:
                    print(f"  Fragment [0, 0, :10]: {tensor[0, 0, :10].tolist()}")
                elif tensor.ndim == 2:
                    print(f"  Fragment [0, :10]: {tensor[0, :10].tolist()}")
                elif tensor.ndim == 1:
                    print(f"  Fragment [:10]: {tensor[:10].tolist()}")
                else:
                    print(
                        f"  Wartość skalarna lub nietypowy wymiar: {tensor.cpu().numpy()}"
                    )

        else:
            print(f"  Typ danych: {type(tensor)}")
            try:
                val_repr = repr(tensor)
                if len(val_repr) > 100:  # Ograniczenie długości dla czytelności
                    val_repr = val_repr[:100] + "..."
                print(f"  Wartość: {val_repr}")
            except Exception as e:
                print(f"  Nie można wyświetlić wartości: {e}")
