"""
POS Tagging en Espanol - Menu Principal
=======================================
Proyecto del curso de Procesamiento de Lenguaje Natural.
Implementa POS Tagging con BiLSTM sobre datasets en espanol.

Uso:
    python main.py
"""

import os
import sys
import torch

from config import (
    ANCORA_PATH, CONLL_TRAIN, CONLL_VALID, CONLL_TEST,
    PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR, REPORTS_DIR,
    ensure_output_dirs,
)


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    print("=" * 60)
    print("   POS TAGGING EN ESPANOL")
    print("   Procesamiento de Lenguaje Natural - Univalle")
    print("=" * 60)


def print_menu():
    print("\n--- MENU PRINCIPAL ---\n")
    print("  [1] Preprocesamiento")
    print("      1a. Preprocesar Ancora")
    print("      1b. Preprocesar CoNLL2002")
    print("      1c. Preprocesar ambos datasets")
    print("      1d. Exportar datos procesados (.pt)")
    print()
    print("  [2] Entrenamiento")
    print("      2a. Entrenar BiLSTM Base (Ancora + CoNLL)")
    print("      2b. Entrenar BiLSTM Deep (Ancora + CoNLL)")
    print("      2c. Entrenar BiLSTM-CRF (Ancora + CoNLL)")
    print()
    print("  [3] Inferencia")
    print("      3a. Etiquetar oraciones (modo interactivo)")
    print("      3b. Ver modelos disponibles")
    print()
    print("  [0] Salir")
    print()


# ── Preprocesamiento ─────────────────────────────────────────

def run_preprocess_ancora():
    print("\n--- Preprocesamiento Ancora ---\n")
    from src.preprocessing.ancora import run_ancora_pipeline, save_ancora_artifacts
    data = run_ancora_pipeline(ANCORA_PATH)
    save_ancora_artifacts(data)


def run_preprocess_conll():
    print("\n--- Preprocesamiento CoNLL2002 ---\n")
    from src.preprocessing.conll import run_conll_pipeline, save_conll_artifacts
    data = run_conll_pipeline(CONLL_TRAIN, CONLL_VALID, CONLL_TEST)
    save_conll_artifacts(data)


def run_preprocess_all():
    print("\n--- Preprocesamiento completo ---\n")
    print("=== ANCORA ===")
    run_preprocess_ancora()
    print("\n=== CONLL2002 ===")
    run_preprocess_conll()
    print("\nFase I completa ejecutada correctamente.")


def run_export():
    print("\n--- Exportar datos procesados ---\n")
    from src.preprocessing.export import main as export_main
    export_main()


# ── Entrenamiento ────────────────────────────────────────────

def check_processed_data():
    """Verifica que existan los datos procesados."""
    ancora_pt = os.path.join(PROCESSED_DIR, "ancora_data.pt")
    conll_pt = os.path.join(PROCESSED_DIR, "conll_data.pt")

    if not os.path.exists(ancora_pt) or not os.path.exists(conll_pt):
        print("\nLos datos procesados no existen.")
        print("Ejecuta primero:")
        print("  1c. Preprocesar ambos datasets")
        print("  1d. Exportar datos procesados (.pt)")
        return False
    return True


def train_bilstm_base():
    print("\n--- Entrenamiento BiLSTM Base ---\n")
    if not check_processed_data():
        return
    _run_training_phase("bilstm_base")


def train_bilstm_deep():
    print("\n--- Entrenamiento BiLSTM Deep ---\n")
    if not check_processed_data():
        return
    _run_training_phase("bilstm_deep")


def train_bilstm_crf():
    print("\n--- Entrenamiento BiLSTM-CRF ---\n")
    if not check_processed_data():
        return
    _run_training_phase("bilstm_crf")


def _run_training_phase(model_type):
    """Ejecuta el entrenamiento de un modelo."""
    from torch.utils.data import Dataset, DataLoader, Subset
    from src.preprocessing.utils import POSDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ensure_output_dirs()

    print("\nCargando datos procesados...")
    ancora = torch.load(os.path.join(PROCESSED_DIR, "ancora_data.pt"), weights_only=False)
    conll = torch.load(os.path.join(PROCESSED_DIR, "conll_data.pt"), weights_only=False)

    train_ds_a = POSDataset(ancora["train_inputs"], ancora["train_labels"], ancora["train_masks"])
    val_ds_a = POSDataset(ancora["val_inputs"], ancora["val_labels"], ancora["val_masks"])
    test_ds_a = POSDataset(ancora["test_inputs"], ancora["test_labels"], ancora["test_masks"])

    train_ds_c = POSDataset(conll["train_inputs"], conll["train_labels"], conll["train_masks"])
    val_ds_c = POSDataset(conll["val_inputs"], conll["val_labels"], conll["val_masks"])
    test_ds_c = POSDataset(conll["test_inputs"], conll["test_labels"], conll["test_masks"])

    print(f"  Ancora: vocab={len(ancora['word2idx'])}, tags={len(ancora['tag2idx'])}, max_len={ancora['max_len']}")
    print(f"  CoNLL2002: vocab={len(conll['word2idx'])}, tags={len(conll['tag2idx'])}, max_len={conll['max_len']}")

    if model_type == "bilstm_base":
        from src.training.bilstm_base.grid_search import run_grid_search
        from src.training.bilstm_base.train import evaluate_model
        from src.training.bilstm_base.predict import print_prediction
        model_file_a = "bilstm_ancora.pt"
        model_file_c = "bilstm_conll.pt"
        model_label = "BiLSTM Base"

    elif model_type == "bilstm_deep":
        from src.training.bilstm_deep.grid_search import run_grid_search
        from src.training.bilstm_deep.train import evaluate_model
        from src.training.bilstm_deep.predict import print_prediction
        model_file_a = "bilstm_deep_ancora.pt"
        model_file_c = "bilstm_deep_conll.pt"
        model_label = "BiLSTM Deep"

    elif model_type == "bilstm_crf":
        from src.training.bilstm_crf.grid_search import run_grid_search
        from src.training.bilstm_crf.train import evaluate_model
        from src.training.bilstm_crf.predict import print_prediction
        model_file_a = "bilstm_crf_ancora.pt"
        model_file_c = "bilstm_crf_conll.pt"
        model_label = "BiLSTM-CRF"

    # --- Grid Search Ancora ---
    print(f"\n{'=' * 60}")
    print(f"GRID SEARCH - ANCORA ({model_label}, 24 combinaciones)")
    print("=" * 60)

    best_model_a, best_config_a, results_a = run_grid_search(
        train_ds_a, val_ds_a,
        vocab_size=len(ancora["word2idx"]),
        tagset_size=len(ancora["tag2idx"]),
        dataset_name="Ancora",
        device=device,
    )

    test_loader_a = DataLoader(test_ds_a, batch_size=64, shuffle=False)
    eval_a = evaluate_model(best_model_a, test_loader_a, ancora["tag2idx"], device)

    # --- Evaluar y guardar Ancora ---
    _eval_save_report(best_model_a, best_config_a, results_a, eval_a,
                      ancora, model_label, model_file_a, "Ancora")

    # --- Grid Search CoNLL2002 ---
    print(f"\n{'=' * 60}")
    print(f"GRID SEARCH - CoNLL2002 ({model_label}, 24 combinaciones)")
    print("=" * 60)

    best_model_c, best_config_c, results_c = run_grid_search(
        train_ds_c, val_ds_c,
        vocab_size=len(conll["word2idx"]),
        tagset_size=len(conll["tag2idx"]),
        dataset_name="CoNLL2002",
        device=device,
    )

    test_loader_c = DataLoader(test_ds_c, batch_size=64, shuffle=False)
    eval_c = evaluate_model(best_model_c, test_loader_c, conll["tag2idx"], device)

    # --- Evaluar y guardar CoNLL ---
    _eval_save_report(best_model_c, best_config_c, results_c, eval_c,
                      conll, model_label, model_file_c, "CoNLL2002")

    # --- Tabla comparativa ---
    _print_comparison_table(model_label, eval_a, eval_c)

    # --- Prediccion de ejemplo ---
    print(f"\n{'=' * 60}")
    print("PREDICCION DE EJEMPLO")
    print("=" * 60)

    test_sentence = "El hombre bajo toca el bajo bajo la escalera"
    print(f'Oracion: "{test_sentence}"')
    print_prediction(test_sentence, best_model_a, ancora["word2idx"], ancora["tag2idx"], device, f"{model_label} - Ancora")
    print_prediction(test_sentence, best_model_c, conll["word2idx"], conll["tag2idx"], device, f"{model_label} - CoNLL2002")

    # --- Tabla global con todos los modelos entrenados ---
    _save_global_comparison_table()


def _eval_save_report(model, config, grid_results, eval_results,
                      data, model_label, model_file, dataset_name):
    """Imprime resultados, guarda modelo .pt y reporte .txt"""
    import pandas as pd

    print(f"\n{'=' * 60}")
    print(f"RESULTADOS TEST - {dataset_name} ({model_label})")
    print(f"{'=' * 60}")
    print(f"Config: {config}")
    print(f"\nAccuracy:    {eval_results['accuracy']:.4f}")
    print(f"Precision:   {eval_results['macro_precision']:.4f}")
    print(f"Recall:      {eval_results['macro_recall']:.4f}")
    print(f"Macro F1:    {eval_results['macro_f1']:.4f}")
    print(f"Weighted F1: {eval_results['weighted_f1']:.4f}")
    print(f"\nClassification Report:\n{eval_results['report']}")

    # Guardar modelo .pt
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_config": config,
        "eval_results": eval_results,
        "grid_results": grid_results,
        "vocab_size": len(data["word2idx"]),
        "tagset_size": len(data["tag2idx"]),
    }, os.path.join(MODELS_DIR, model_file))
    print(f"Modelo guardado en {model_file}")

    # Guardar reporte .txt
    report_name = model_file.replace(".pt", "_report.txt")
    report_path = os.path.join(REPORTS_DIR, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"REPORTE - {model_label} / {dataset_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Mejor configuracion: {config}\n\n")
        f.write("METRICAS GLOBALES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:    {eval_results['accuracy']:.4f}\n")
        f.write(f"Precision:   {eval_results['macro_precision']:.4f}\n")
        f.write(f"Recall:      {eval_results['macro_recall']:.4f}\n")
        f.write(f"Macro F1:    {eval_results['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {eval_results['weighted_f1']:.4f}\n\n")
        f.write("METRICAS POR ETIQUETA\n")
        f.write("-" * 40 + "\n")
        f.write(eval_results['report'] + "\n\n")
        f.write("DETALLE GRID SEARCH (ordenado por val_loss)\n")
        f.write("-" * 40 + "\n")
        df_grid = pd.DataFrame(grid_results).sort_values("best_val_loss")
        f.write(df_grid.to_string(index=False) + "\n")
    print(f"Reporte guardado en outputs/reports/{report_name}")


def _print_comparison_table(model_label, eval_a, eval_c):
    """Imprime tabla comparativa del modelo actual."""
    import pandas as pd

    print(f"\n{'=' * 80}")
    print(f"TABLA DE RESULTADOS - {model_label}")
    print("=" * 80)

    table = pd.DataFrame([
        {
            "Modelo": model_label, "Dataset": "Ancora",
            "Accuracy": f"{eval_a['accuracy']:.4f}",
            "Precision": f"{eval_a['macro_precision']:.4f}",
            "Recall": f"{eval_a['macro_recall']:.4f}",
            "F1-Score": f"{eval_a['macro_f1']:.4f}",
        },
        {
            "Modelo": model_label, "Dataset": "CoNLL2002",
            "Accuracy": f"{eval_c['accuracy']:.4f}",
            "Precision": f"{eval_c['macro_precision']:.4f}",
            "Recall": f"{eval_c['macro_recall']:.4f}",
            "F1-Score": f"{eval_c['macro_f1']:.4f}",
        }
    ])
    print(table.to_string(index=False))


def _save_global_comparison_table():
    """Genera tabla comparativa global con todos los modelos entrenados."""
    import pandas as pd
    from src.inference.tagger import list_available_models

    available = list_available_models()
    if not available:
        return

    rows = []
    label_map = {
        "bilstm_base": "BiLSTM",
        "bilstm_deep": "BiLSTM+Dense",
        "bilstm_crf": "BiLSTM-CRF",
    }
    dataset_map = {"ancora": "Ancora", "conll": "CoNLL2002"}

    for model_type, dataset_name, filename in available:
        checkpoint = torch.load(
            os.path.join(MODELS_DIR, filename), map_location="cpu", weights_only=False
        )
        ev = checkpoint.get("eval_results", {})
        rows.append({
            "Modelo": label_map.get(model_type, model_type),
            "Dataset": dataset_map.get(dataset_name, dataset_name),
            "Accuracy": f"{ev.get('accuracy', 0):.4f}",
            "Precision": f"{ev.get('macro_precision', 0):.4f}",
            "Recall": f"{ev.get('macro_recall', 0):.4f}",
            "F1-Score": f"{ev.get('macro_f1', 0):.4f}",
        })

    if not rows:
        return

    table = pd.DataFrame(rows)

    print(f"\n{'=' * 80}")
    print("TABLA COMPARATIVA GLOBAL - TODOS LOS MODELOS")
    print("=" * 80)
    print(table.to_string(index=False))

    # Guardar en archivo
    table_path = os.path.join(REPORTS_DIR, "tabla_comparativa_global.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("TABLA COMPARATIVA GLOBAL - POS TAGGING\n")
        f.write("=" * 80 + "\n\n")
        f.write(table.to_string(index=False) + "\n")
    print(f"\nTabla guardada en outputs/reports/tabla_comparativa_global.txt")


# ── Inferencia ───────────────────────────────────────────────

def run_interactive_tagging():
    from src.inference.tagger import interactive_tagging
    interactive_tagging()


def show_available_models():
    from src.inference.tagger import list_available_models
    available = list_available_models()

    print("\n--- Modelos entrenados disponibles ---\n")
    if not available:
        print("  No hay modelos entrenados.")
        print("  Ejecuta primero el entrenamiento desde el menu.")
    else:
        for mtype, dset, fname in available:
            path = os.path.join(MODELS_DIR, fname)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  - {mtype:15s} | {dset:8s} | {fname} ({size_mb:.1f} MB)")


# ── Menu principal ───────────────────────────────────────────

def main():
    ensure_output_dirs()

    while True:
        print_header()
        print_menu()

        try:
            option = input("Opcion: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo...")
            break

        if option == "0":
            print("\nSaliendo...")
            break

        actions = {
            "1a": run_preprocess_ancora,
            "1b": run_preprocess_conll,
            "1c": run_preprocess_all,
            "1d": run_export,
            "2a": train_bilstm_base,
            "2b": train_bilstm_deep,
            "2c": train_bilstm_crf,
            "3a": run_interactive_tagging,
            "3b": show_available_models,
        }

        action = actions.get(option)
        if action:
            try:
                action()
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nOpcion no valida. Intenta de nuevo.")

        input("\nPresiona Enter para continuar...")
        clear_screen()


if __name__ == "__main__":
    main()
