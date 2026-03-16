import copy
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import BiLSTMTagger
from train import train_model


def run_grid_search(train_ds, val_ds, vocab_size, tagset_size, dataset_name, device):
    """Ejecuta grid search sobre todas las combinaciones de hiperparámetros."""

    batch_sizes = [16, 32, 64]
    optimizers_config = [
        {"name": "Adam", "lr": 0.001},
        {"name": "SGD", "lr": 0.01, "momentum": 0.9}
    ]
    embedding_dims = [100, 300]
    hidden_dims = [128, 256]

    results = []
    best_val_loss = float('inf')
    best_model = None
    best_config = None

    total = len(batch_sizes) * len(optimizers_config) * len(embedding_dims) * len(hidden_dims)
    combo = 0

    for bs in batch_sizes:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

        for opt_cfg in optimizers_config:
            for emb_dim in embedding_dims:
                for hid_dim in hidden_dims:
                    combo += 1
                    print(f"\n[{dataset_name}] {combo}/{total}: bs={bs}, opt={opt_cfg['name']}, emb={emb_dim}, hid={hid_dim}")

                    model = BiLSTMTagger(vocab_size, tagset_size, emb_dim, hid_dim)

                    if opt_cfg["name"] == "Adam":
                        optimizer = optim.Adam(model.parameters(), lr=opt_cfg["lr"])
                    else:
                        optimizer = optim.SGD(model.parameters(), lr=opt_cfg["lr"], momentum=opt_cfg["momentum"])

                    start = time.time()
                    trained_model, t_losses, v_losses, val_loss = train_model(
                        model, train_loader, val_loader, optimizer, device, epochs=50, patience=5
                    )
                    elapsed = time.time() - start

                    config = {
                        "batch_size": bs,
                        "optimizer": opt_cfg["name"],
                        "embedding_dim": emb_dim,
                        "hidden_dim": hid_dim,
                        "best_val_loss": round(val_loss, 4),
                        "epochs_run": len(t_losses),
                        "time_s": round(elapsed, 1)
                    }
                    results.append(config)

                    print(f"  -> Val Loss: {val_loss:.4f} | Epochs: {len(t_losses)} | Tiempo: {elapsed:.1f}s")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = copy.deepcopy(trained_model)
                        best_config = config.copy()

                    del model, trained_model, optimizer
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print(f"MEJOR CONFIG [{dataset_name}]: {best_config}")
    return best_model, best_config, results
