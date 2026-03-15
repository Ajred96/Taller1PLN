from run_phase1_ancora import main as run_ancora
from run_phase1_conll import main as run_conll


def main():
    print("Ejecutando Fase I completa...\n")

    print("=== ANCORA ===")
    run_ancora()

    print("\n=== CONLL2002 ===")
    run_conll()

    print("\nFase I completa ejecutada correctamente.")
    print("Revisa outputs/artifacts y outputs/reports.")


if __name__ == "__main__":
    main()