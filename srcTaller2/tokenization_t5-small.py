import json
import os
import pandas as pd
from transformers import AutoTokenizer

def cargar_primeras_oraciones(ruta_json):
    """Carga el archivo JSON con las 3 primeras oraciones de cada split."""
    if not os.path.exists(ruta_json):
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta_json}")
    with open(ruta_json, 'r', encoding='utf-8') as f:
        return json.load(f)

def procesar_con_sentencepiece(dataset_name, datos_splits):
    """
    Aplica el tokenizador SentencePiece de t5-small sobre las oraciones de cada split
    y extrae las métricas analíticas solicitadas en el taller.
    """
    print(f"📦 Cargando tokenizador SentencePiece (google-t5/t5-small) para {dataset_name}...")
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    
    registros_sp = []
    
    for split_name, oraciones in datos_splits.items():
        for idx, lista_tokens in enumerate(oraciones):
            # Reconstrucción: Convertimos la lista de tokens en texto continuo 
            texto_original = " ".join(lista_tokens)
            
            # Tokenización real con SentencePiece
            tokens_sp = tokenizer.tokenize(texto_original)
            
            # --- Extracción de Métricas para comparar WordPiece y SentencePiece ---
            total_tokens = len(tokens_sp)
            
            # Identificamos subpalabras fragmentadas (en T5, si NO empieza con ' ', es subpalabra)
            fragmentos = [
                t for t in tokens_sp 
                if not t.startswith(" ") and t not in [".", ",", "(", ")", '"', "-", "##"]
            ]
            
            # Tratamiento de palabras desconocidas (OOV) -> Busca el token <unk>
            tiene_oov = "<unk>" in tokens_sp
            
            registros_sp.append({
                "Dataset": dataset_name,
                "Split": split_name,
                "ID_Frase": idx,
                "Texto_Original": texto_original,
                "Tokens_SentencePiece": tokens_sp,
                "Total_Tokens_SP": total_tokens,
                "Cant_Fragmentos_SP": len(fragmentos),
                "Fragmentos_Detectados_SP": fragmentos,
                "¿Tiene_OOV_SP?": "Sí" if tiene_oov else "No"
            })
            
            print(f"\n✨ [{dataset_name} - {split_name.upper()}] Frase #{idx}")
            print(f"📝 Original: {texto_original}")
            print(f"🧩 Tokens SP: {tokens_sp}")
            print(f"📊 Métricas: Total Tokens = {total_tokens} | Subwords Fragmentadas = {len(fragmentos)} | ¿OOV? = {'Sí' if tiene_oov else 'No'}")
            print("-" * 80)
            
    return registros_sp

if __name__ == "__main__":
    # Define la ruta exacta donde el Integrante 1 te dejó los archivos
    base_path = r"D:\Univalle\2026-1\PLN\Taller1PLN\srcTaller2\outputs\processed"
    
    ruta_conll = os.path.join(base_path, "conll2002_first_3.json")
    ruta_ancora = os.path.join(base_path, "ancora_first_3.json")
    
    try:
        datos_conll = cargar_primeras_oraciones(ruta_conll)
        datos_ancora = cargar_primeras_oraciones(ruta_ancora)
        
        resultados_conll_sp = procesar_con_sentencepiece("CoNLL2002", datos_conll)
        resultados_ancora_sp = procesar_con_sentencepiece("AnCora", datos_ancora)
        
        todos_mis_resultados = resultados_conll_sp + resultados_ancora_sp
        df_mis_metricas = pd.DataFrame(todos_mis_resultados)
        
        output_csv = os.path.join(base_path, "mis_resultados_sentencepiece.csv")
        df_mis_metricas.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n💾 ¡PROCESO TERMINADO! Tus métricas de SentencePiece se guardaron en:\n👉 {output_csv}")
        
    except Exception as e:
        print(f"\n❌ Ocurrió un error al ejecutar el proceso: {e}")