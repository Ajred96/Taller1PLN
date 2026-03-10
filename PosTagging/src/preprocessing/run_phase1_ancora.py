import json
import os

from build_dataloaders import build_ancora_dataloaders


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

    path = "../../data/raw/ancora/ancora_corpus_pos.csv"
    data = build_ancora_dataloaders(path, batch_size=32)

    save_json(data["word2idx"], "../../outputs/artifacts/ancora_word2idx.json")
    save_json(data["tag2idx"], "../../outputs/artifacts/ancora_tag2idx.json")

    summary = []
    summary.append("FASE I - RESUMEN ANCORA")
    summary.append("=" * 40)
    summary.append("Dataset: Ancora")
    summary.append(f"Word vocab size: {len(data['word2idx'])}")
    summary.append(f"Tag vocab size: {len(data['tag2idx'])}")
    summary.append(f"Max length: {data['max_len']}")
    summary.append("Batch size: 32")
    summary.append("")
    summary.append("Artefactos generados:")
    summary.append("- outputs/artifacts/ancora_word2idx.json")
    summary.append("- outputs/artifacts/ancora_tag2idx.json")
    summary.append("")
    summary.append("Objetos disponibles para entrenamiento:")
    summary.append("- train_loader")
    summary.append("- val_loader")
    summary.append("- test_loader")
    summary.append("- word2idx")
    summary.append("- tag2idx")
    summary.append("- max_len")

    save_text("\n".join(summary), "../../outputs/reports/phase1_ancora_summary.txt")

    print("\nAncora procesado correctamente.")
    print("Se guardaron vocabularios y resumen.")


if __name__ == "__main__":
    main()
