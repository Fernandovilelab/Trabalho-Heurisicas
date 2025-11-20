import pandas as pd
import argparse

# ==========================
# PARÂMETROS VIA EXECUÇÃO
# ==========================
parser = argparse.ArgumentParser(description="Gerador automático de precedências geométricas (9 vizinhos acima).")

parser.add_argument(
    "--arq_blocos",
    type=str,
    default="Modelo de Blocos.csv",
    help="Arquivo de entrada contendo o modelo de blocos (com colunas id,x,y,z)."
)

parser.add_argument(
    "--arq_saida",
    type=str,
    default="Modelo_com_Precedencias.csv",
    help="Arquivo CSV de saída com as precedências geradas."
)

args = parser.parse_args()

ARQ_INPUT = args.arq_blocos
ARQ_OUT   = args.arq_saida

df = pd.read_csv(ARQ_INPUT, sep=';')


# Cria DataFrame auxiliar com as 9 coordenadas precedentes
precedence = df[["x", "y", "z"]].copy()
precedence['prec1'] = list(zip(precedence['x'],     precedence['y'],     precedence['z'] + 1))
precedence['prec2'] = list(zip(precedence['x']-1,   precedence['y']+1,   precedence['z'] + 1))
precedence['prec3'] = list(zip(precedence['x']-1,   precedence['y'],     precedence['z'] + 1))
precedence['prec4'] = list(zip(precedence['x']-1,   precedence['y']-1,   precedence['z'] + 1))
precedence['prec5'] = list(zip(precedence['x'],     precedence['y']+1,   precedence['z'] + 1))
precedence['prec6'] = list(zip(precedence['x'],     precedence['y']-1,   precedence['z'] + 1))
precedence['prec7'] = list(zip(precedence['x']+1,   precedence['y']+1,   precedence['z'] + 1))
precedence['prec8'] = list(zip(precedence['x']+1,   precedence['y'],     precedence['z'] + 1))
precedence['prec9'] = list(zip(precedence['x']+1,   precedence['y']-1,   precedence['z'] + 1))

# Agora substitui as coordenadas pelos IDs reais dos blocos precedentes (se existirem)
coord2id = {(x, y, z): i for i, x, y, z in df[["id", "x", "y", "z"]].itertuples(index=False)}

for i in range(1, 10):
    precedence[f'prec{i}'] = precedence[f'prec{i}'].map(coord2id).fillna(-1).astype(int)

# Junta com os IDs originais
df_out = pd.concat([df[["id"]], precedence[[f'prec{i}' for i in range(1, 10)]]], axis=1)

# === Salva resultado ===
df_out.to_csv(ARQ_OUT, index=False)
print(df_out.head())


