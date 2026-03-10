import json
import os

from prepare_conll_pipeline import prepare_conll_pipeline


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_text(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def ensure_dirs():
    os.makedirs("../../outputs/artifacts", exist_ok=True)
    os.makedirs("../../outputs/reports", exist_ok=True)


def main():
    ensure_dirs()

    train_path = "../../data/raw/conll2002/train.txt"
    valid_path = "../../data/raw/conll2002/valid.txt"
    test_path = "../../data/raw/conll2002/test.txt"

    data = prepare_conll_pipeline(train_path, valid_path, test_path, batch_size=32)

    save_json(data["word2idx"], "../../outputs/artifacts/conll_word2idx.json")
    save_json(data["tag2idx"], "../../outputs/artifacts/conll_tag2idx.json")

    summary = []
    summary.append("FASE I - RESUMEN CONLL2002")
    summary.append("=" * 40)
    summary.append("Dataset: CoNLL2002")
    summary.append(f"Train sentences: {len(data['X_train'])}")
    summary.append(f"Validation sentences: {len(data['X_val'])}")
    summary.append(f"Test sentences: {len(data['X_test'])}")
    summary.append(f"Word vocab size: {len(data['word2idx'])}")
    summary.append(f"Tag vocab size: {len(data['tag2idx'])}")
    summary.append(f"Max length: {data['max_len']}")
    summary.append("Batch size: 32")
    summary.append("")
    summary.append("Observación importante:")
    summary.append("- CoNLL2002 tiene oraciones extremadamente largas.")
    summary.append("- max_len observado en train: 1238")
    summary.append("- Conviene revisar truncamiento o percentiles antes del entrenamiento.")
    summary.append("")
    summary.append("Artefactos generados:")
    summary.append("- outputs/artifacts/conll_word2idx.json")
    summary.append("- outputs/artifacts/conll_tag2idx.json")
    summary.append("")
    summary.append("Objetos disponibles para entrenamiento:")
    summary.append("- train_loader")
    summary.append("- val_loader")
    summary.append("- test_loader")
    summary.append("- word2idx")
    summary.append("- tag2idx")
    summary.append("- max_len")

    save_text("\n".join(summary), "../../outputs/reports/phase1_conll_summary.txt")

    print("\nCoNLL2002 procesado correctamente.")
    print("Se guardaron vocabularios y resumen.")


if __name__ == "__main__":
    main()