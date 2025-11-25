#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRASP para CPIT — compatível com seu baseline_toposort.py
- Entrada blocks (sep=';') e precedences (csv padrão com colunas prec*)
- Regras idênticas ao baseline (custo lavra, capacidade de processo, estéril ilimitado, desconto)
- Saídas:
   grasp_best_blocks.csv
   grasp_best_summary.csv
"""

import pandas as pd
import argparse
import random
from copy import deepcopy
from collections import defaultdict

# ---------------------------
# argumentos
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--arq_blocos", type=str, default="kd_block_model.csv",
                    help="Arquivo de blocos (sep=';')")
parser.add_argument("--arq_precs", type=str, default="kd_precedence.csv",
                    help="Arquivo com precedências (csv padrão)")
parser.add_argument("--arq_out_blocks", type=str, default="grasp_best_blocks.csv",
                    help="Arquivo de saída: blocos ordenados (melhor)")
parser.add_argument("--arq_out_sum", type=str, default="grasp_best_summary.csv",
                    help="Arquivo de saída: sumário anual (melhor)")
parser.add_argument("--proc_cap_tpy", type=float, default=10_000_000,
                    help="Capacidade de processamento (t/ano)")
parser.add_argument("--discount", type=float, default=0.15,
                    help="Taxa de desconto")
parser.add_argument("--anos", type=int, default=9,
                    help="Número de anos do modelo")
parser.add_argument("--iterations", type=int, default=50,
                    help="Número de iterações GRASP")
parser.add_argument("--alpha", type=float, default=0.25,
                    help="Alpha GRASP (0 = guloso, 1 = aleatório)")
args = parser.parse_args()

PROC_CAP_TPY = args.proc_cap_tpy
DISCOUNT = args.discount
ANOS = args.anos
ITER = args.iterations
ALPHA = args.alpha

# ---------------------------
# Leitura dados
# ---------------------------
def load_blocks(blocks_csv, prec_csv):
    df = pd.read_csv(blocks_csv, sep=';')
    df["id"] = df["id"].astype(int)
    df["z"] = df["z"].astype(int)
    df["destination"] = df["destination"].astype(int)
    df["tonn"] = df["tonn"].astype(float)
    # process_profit pode já existir
    if "process_profit" not in df.columns:
        df["process_profit"] = 0.0
    df["process_profit"] = df["process_profit"].astype(float)

    # economic values (mesma lógica do baseline)
    df["val_ore"] = -0.75 * df["tonn"] + df["process_profit"]
    df["val_waste"] = -0.75 * df["tonn"]
    df["vpt"] = df["val_ore"] / df["tonn"].clip(lower=1e-9)

    # precedences
    prec_df = pd.read_csv(prec_csv)
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

    # ensure every block has an entry
    for bid in df["id"].tolist():
        precedences.setdefault(int(bid), [])

    return df, precedences

# ---------------------------
# Helpers similares ao baseline
# ---------------------------
def build_succs(precedences, all_ids):
    succs = defaultdict(list)
    for b, ps in precedences.items():
        for p in ps:
            if p in all_ids:
                succs[p].append(b)
    return succs

# score para GRASP (você pode ajustar)
def block_score(df_indexed, bid):
    # Para minério usamos vpt (valor por tonelada) e preferir z mais alto
    dest = int(df_indexed.at[bid, "destination"])
    if dest == 1:
        # combine vpt e z to break ties
        return float(df_indexed.at[bid, "vpt"]) + 1e-6 * (1000 - df_indexed.at[bid, "z"])
    else:
        # estéril: use potencial de destrave (valores baixos por si mesmos)
        return float(df_indexed.at[bid, "val_waste"])

# ---------------------------
# Construção GRASP (topológica)
# ---------------------------
def grasp_construct(df_blocks, precedences, alpha=0.25, seed=None):
    if seed is not None:
        random.seed(seed)
    all_ids = set(df_blocks["id"].tolist())
    remaining_preds = {b: len(precedences.get(b, [])) for b in all_ids}
    eligible = [b for b, cnt in remaining_preds.items() if cnt == 0]
    df_idx = df_blocks.set_index("id")

    succs = build_succs(precedences, all_ids)
    solution = []

    while eligible:
        # calc scores
        scores = {b: block_score(df_idx, b) for b in eligible}
        # sort by score desc
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        max_score = sorted_scores[0][1]
        min_score = sorted_scores[-1][1]
        threshold = max_score - alpha * (max_score - min_score)

        # RCL
        RCL = [b for b, sc in sorted_scores if sc >= threshold]
        chosen = random.choice(RCL)
        solution.append(chosen)

        # update eligible
        eligible.remove(chosen)
        for child in succs.get(chosen, []):
            remaining_preds[child] -= 1
            if remaining_preds[child] == 0:
                eligible.append(child)

    return solution

# ---------------------------
# Avaliação: simula anos (mesma lógica do baseline)
# ---------------------------
def evaluate_sequence(sequence, df_blocks, precedences,
                      proc_cap=PROC_CAP_TPY, discount=DISCOUNT, anos=ANOS):
    """
    Recebe uma sequence (lista de ids topológica válida) e simula ano a ano
    usando as mesmas regras do baseline_toposort.
    Retorna: vpl_total, schedule_df, summary_df
    """
    all_ids = set(df_blocks["id"].tolist())
    tonn = dict(zip(df_blocks["id"], df_blocks["tonn"]))
    dest = dict(zip(df_blocks["id"], df_blocks["destination"]))
    zlev = dict(zip(df_blocks["id"], df_blocks["z"]))
    val_ore = dict(zip(df_blocks["id"], df_blocks["val_ore"]))
    val_waste = dict(zip(df_blocks["id"], df_blocks["val_waste"]))

    succs = build_succs(precedences, all_ids)
    # Predecessors helper (filter to ids present)
    def preds_of(bid):
        return [p for p in precedences.get(bid, []) if p in all_ids]

    mined = set()
    rows = []
    seq_set = set(sequence)
    # ensure sequence contains exactly the blocks we have (or subset)
    # We'll follow sequence order but only allow blocks whose predecessors are mined.
    # For each year, scan sequence in order and place eligible blocks until no more can be placed.
    for year in range(1, anos + 1):
        remaining = proc_cap
        placed_any_year = False
        guard = 0
        while True:
            guard += 1
            if guard > 2_000_000:
                break

            placed_this_loop = False
            # iterate sequence in order
            for b in sequence:
                if b in mined:
                    continue
                # only consider blocks that exist in our set
                if b not in all_ids:
                    continue
                # check predecessors
                ps = preds_of(b)
                if any(p not in mined for p in ps):
                    continue  # not yet eligible
                # eligible now
                if dest[b] == 1:
                    t = tonn[b]
                    if t <= remaining + 1e-9:
                        cash = val_ore[b]
                        disc = cash / ((1 + discount) ** (year - 1))
                        rows.append({"id": b, "year": year, "dest": 1, "tonn": t,
                                     "cashflow": cash, "discounted": disc, "z": zlev[b]})
                        mined.add(b)
                        placed_this_loop = True
                        remaining -= t
                    else:
                        # can't fit this ore now, try later (don't force year change here)
                        continue
                else:
                    # waste: always mine (doesn't consume process capacity)
                    cash = val_waste[b]
                    disc = cash / ((1 + discount) ** (year - 1))
                    rows.append({"id": b, "year": year, "dest": 2, "tonn": tonn[b],
                                 "cashflow": cash, "discounted": disc, "z": zlev[b]})
                    mined.add(b)
                    placed_this_loop = True
                # after mining a block, we don't immediately update sequence order because sequence order remains
                # continue scanning to try to fill year
            if not placed_this_loop:
                break
            placed_any_year = True
            # try again to fill remaining capacity in the same year
        # end while for year
        # debug print optional:
        # print(f"[GRASP eval] Ano {year}: {len(mined)} blocos lavrados, {remaining:,.0f} ton restantes.")
        # If no blocks were placed this year, we can't proceed further (no eligible blocks)
        if not placed_any_year:
            # stop early — nothing more can be mined inside ANOS
            break

    if not rows:
        # nada minerado dentro dos anos
        return float("-inf"), pd.DataFrame(), pd.DataFrame()

    sched = pd.DataFrame(rows).sort_values(["year", "z"], ascending=[True, False])
    # summary
    res_list = []
    for y, g in sched.groupby("year"):
        ore = g.loc[g["dest"] == 1, "tonn"].sum()
        waste = g.loc[g["dest"] == 2, "tonn"].sum()
        cf = g["cashflow"].sum()
        dfc = g["discounted"].sum()
        res_list.append({"year": int(y), "ore_tonn": float(ore), "waste_tonn": float(waste),
                         "cashflow": float(cf), "discounted": float(dfc)})
    res = pd.DataFrame(res_list).sort_values("year").reset_index(drop=True)
    res["cum_discounted"] = res["discounted"].cumsum()
    vpl_total = float(res["discounted"].sum())

    return vpl_total, sched, res

# ---------------------------
# Loop GRASP
# ---------------------------
def grasp_search(df_blocks, precedences, iterations=50, alpha=0.25):
    best_vpl = float("-inf")
    best_sched = None
    best_res = None
    best_seq = None

    for it in range(1, iterations + 1):
        seq = grasp_construct(df_blocks, precedences, alpha=alpha, seed=None)
        vpl, sched, res = evaluate_sequence(seq, df_blocks, precedences)
        print(f"Iter {it}/{iterations} -> VPL = {vpl:.2f}")
        if vpl > best_vpl:
            best_vpl = vpl
            best_sched = sched
            best_res = res
            best_seq = seq

    return best_seq, best_vpl, best_sched, best_res

# ---------------------------
# main
# ---------------------------
if __name__ == "__main__":
    df_blocks, precedences = load_blocks(args.arq_blocos, args.arq_precs)
    best_seq, best_vpl, best_sched, best_res = grasp_search(df_blocks, precedences,
                                                            iterations=ITER, alpha=ALPHA)
    if best_sched is None:
        print("Nenhuma solução válida foi encontrada dentro dos parâmetros.")
    else:
        best_sched.to_csv(args.arq_out_blocks, index=False)
        best_res.to_csv(args.arq_out_sum, index=False)
        print("\n✅ GRASP finalizado")
        print(f"Melhor VPL = {best_vpl:,.2f}")
        print(f"Saídas: {args.arq_out_blocks}, {args.arq_out_sum}")
