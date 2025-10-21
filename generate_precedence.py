import pandas as pd

df = pd.read_csv("Modelo de Blocos.csv",sep=';')


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
df_out.to_csv("Modelo_com_Precedencias.csv", index=False)
print(df_out.head())


