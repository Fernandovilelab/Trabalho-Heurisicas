# -*- coding: utf-8 -*-
"""
TopoSort destravante (sem PuLP) — 9 anos
Regras:
- Destinations: 1=ore, 2=waste
- Process capacity: 10 Mt/ano (apenas minério)
- Mineração de estéril: ilimitada (sempre respeitando precedência)
- Custo de lavra: -0.75 * tonn
- blockvalue (minério) = -0.75*tonn + process_profit
- Desconto: 0.15
Entradas:
  - "Modelo de Blocos.csv" (sep=';')
  - "Modelo_com_Precedencias.csv" (CSV com id, prec1..prec9; -1 = sem predecessor)
Saídas:
  - "toposort_baseline_blocks.csv"
  - "toposort_baseline_summary.csv"
Mostra no final o VPL total.
"""

import pandas as pd
from collections import defaultdict

# ========= CONFIG =========
ARQ_BLOCOS = "Modelo de Blocos.csv"          # usa ';'
ARQ_PRECS  = "Modelo_com_Precedencias.csv"   # padrão ','
ARQ_OUT_BLOCKS = "toposort_baseline_blocks.csv"
ARQ_OUT_SUM    = "toposort_baseline_summary.csv"

PROC_CAP_TPY = 10_000_000   # 10 Mt/ano (só minério)
DISCOUNT = 0.15
ANOS = 9                    # Horizonte igual ao do site

# ========= LEITURA =========
df = pd.read_csv(ARQ_BLOCOS, sep=';')
df["id"] = df["id"].astype(int)
df["z"] = df["z"].astype(int)
df["destination"] = df["destination"].astype(int)
df["tonn"] = df["tonn"].astype(float)
df["process_profit"] = df["process_profit"].astype(float)

# Valores econômicos
df["val_ore"]   = -0.75 * df["tonn"] + df["process_profit"]
df["val_waste"] = -0.75 * df["tonn"]
# Para ordenação do minério usaremos valor por tonelada (mais “LP-like”)
df["vpt"] = df["val_ore"] / df["tonn"].clip(lower=1e-9)

# Precedências
prec_df = pd.read_csv(ARQ_PRECS)
prec_cols = [c for c in prec_df.columns if c.lower().startswith("prec")]
precedences = {}
for _, r in prec_df.iterrows():
    preds = []
    for c in prec_cols:
        v = r[c]
        if pd.isna(v): 
            continue
        try:
            v = int(v)
        except Exception:
            continue
        if v != -1:
            preds.append(v)
    precedences[int(r["id"])] = preds

all_ids = set(df["id"])

# Sucessores
succs = defaultdict(list)
for b, ps in precedences.items():
    for p in ps:
        if p in all_ids:
            succs[p].append(b)

# Dicionários rápidos
tonn = dict(zip(df["id"], df["tonn"]))
dest = dict(zip(df["id"], df["destination"]))
zlev = dict(zip(df["id"], df["z"]))
val_ore = dict(zip(df["id"], df["val_ore"]))
val_waste = dict(zip(df["id"], df["val_waste"]))
vpt = dict(zip(df["id"], df["vpt"]))

def preds_of(bid):
    return [p for p in precedences.get(bid, []) if p in all_ids]

# ========= HELPERS =========
def all_preds_mined(bid, mined):
    ps = preds_of(bid)
    return all(p in mined for p in ps)

def eligible(mined):
    rem = [i for i in all_ids if i not in mined]
    elig = [i for i in rem if all_preds_mined(i, mined)]
    ores   = [i for i in elig if dest[i] == 1]
    wastes = [i for i in elig if dest[i] == 2]
    return ores, wastes

def missing_preds(bid, mined):
    return [p for p in preds_of(bid) if p not in mined]

def waste_unlock_score(wid, mined):
    """
    Score para escolher estéril quando não há minério elegível/ suficiente.
    Para cada filho c (de minério) ainda travado:
      soma max(0, val_ore[c]) / faltantes(c)
    Preferimos estéril que aproxima minérios valiosos e quase livres.
    """
    sc = 0.0
    for c in succs.get(wid, []):
        if dest.get(c, 2) != 1:
            continue
        miss = missing_preds(c, mined)
        if miss and wid in miss:
            sc += max(0.0, val_ore.get(c, 0.0)) / float(len(miss))
    return sc

# ========= HEURÍSTICA: por ano tentar ENCHER 10 Mt =========
mined = set()
rows = []

for year in range(1, ANOS + 1):
    remaining = PROC_CAP_TPY
    mined_count_before = len(mined)

    # Loop até encher a capacidade ou não haver mais como avançar
    guard = 0
    while remaining > 1e-6:   # sobra marginal ignora
        guard += 1
        if guard > 1_000_000:
            # segurança
            break

        ores, wastes = eligible(mined)

        # 1) Se houver minério elegível, tente usar a capacidade no melhor por vpt
        if ores:
            # Ordena por valor/ton (desc), depois por z (mais alto primeiro), depois por id
            ores_sorted = sorted(ores, key=lambda b: (vpt[b], -zlev[b], -val_ore[b], -tonn[b], -b), reverse=True)
            placed_any = False

            for b in ores_sorted:
                t = tonn[b]
                if t <= remaining + 1e-9:
                    cash = val_ore[b]
                    disc = cash / ((1 + DISCOUNT) ** (year - 1))
                    rows.append({"id": b, "year": year, "dest": 1, "tonn": t,
                                 "cashflow": cash, "discounted": disc, "z": zlev[b]})
                    mined.add(b)
                    remaining -= t
                    placed_any = True
                    # continue tentando encher
            if placed_any:
                continue  # volta e tenta colocar mais minério neste ano

            # Há minério elegível mas nenhum cabe na capacidade restante -> tentar abrir mais minério com estéril
            # (ou parar se não houver estéril elegível)
        
        # 2) Se não conseguimos colocar minério (não existe ou não cabe), lavrar estéril elegível “mais destravante”
        if wastes:
            # escolhe pelo maior unlock score; em empates, menor z (mais alto) e menor id
            best_w = None
            best_sc = -1.0
            for w in wastes:
                sc = waste_unlock_score(w, mined)
                if (sc > best_sc) or (abs(sc - best_sc) < 1e-12 and (zlev[w] < (zlev.get(best_w, 10**9))) ) or (abs(sc - best_sc) < 1e-12 and zlev[w] == zlev.get(best_w, 10**9) and w < (best_w or 10**9)):
                    best_sc = sc
                    best_w = w
            if best_w is None:
                # fallback improvável
                best_w = sorted(wastes, key=lambda w: (zlev[w], w))[0]

            # lavra estéril (não consome capacidade de processo)
            cash = val_waste[best_w]
            disc = cash / ((1 + DISCOUNT) ** (year - 1))
            rows.append({"id": best_w, "year": year, "dest": 2, "tonn": tonn[best_w],
                         "cashflow": cash, "discounted": disc, "z": zlev[best_w]})
            mined.add(best_w)
            # volta ao loop: isso pode liberar novos minérios OU permitir que algum minério que não cabia seja trocado por outro que caiba
            continue

        # 3) Chegamos aqui: sem minério que caiba e sem estéril elegível -> não há como avançar neste ano
        break

    print(f"Ano {year}: {len(mined)} blocos lavrados, {remaining:,.0f} ton restantes.")

# ========= RESULTADOS =========
if not rows:
    raise RuntimeError("Nenhum bloco lavrado. Verifique arquivos de entrada.")

sched = pd.DataFrame(rows).sort_values(["year", "z"], ascending=[True, False])
sched.to_csv(ARQ_OUT_BLOCKS, index=False)

# Resumo anual
res_list = []
for y, g in sched.groupby("year"):
    ore = g.loc[g["dest"] == 1, "tonn"].sum()
    waste = g.loc[g["dest"] == 2, "tonn"].sum()
    cf = g["cashflow"].sum()
    dfc = g["discounted"].sum()
    res_list.append({"year": y, "ore_tonn": ore, "waste_tonn": waste,
                     "cashflow": cf, "discounted": dfc})
res = pd.DataFrame(res_list).sort_values("year").reset_index(drop=True)
res["cum_discounted"] = res["discounted"].cumsum()
res.to_csv(ARQ_OUT_SUM, index=False)

vpl_total = float(res["discounted"].sum())
print("\n✅ TopoSort manual concluído!")
print(f"- {ARQ_OUT_BLOCKS}")
print(f"- {ARQ_OUT_SUM}")
print(f"\n💰 Valor total obtido (VPL) = {vpl_total:,.2f}")
