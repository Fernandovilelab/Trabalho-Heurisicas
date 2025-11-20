Para gerar o modelo de precedencia (restrição geografica)
python generate_precedence.py --arq_blocos Modelo_de_Blocos.csv --arq_saida block_precedence.csv

Para rodar o baseline toposort:
Arquivos necessários

Modelo de Blocos.csv — modelo de blocos (separador ;)

Modelo_com_Precedencias.csv — relações de precedência (separador ,)

  python3 baseline_toposort.py --arq_blocos Modelo_de_Blocos.csv --arq_precs Modelo_com_Precedencias.csv --arq_out_blocks blocos_ordenados.csv --arq_out_sum resumo.csv --proc_cap_tpy 12000000 --discount 0.12  --anos 12


Para rodar o algoritmo genético:

  python3 merge_precedences.py "Modelo de Blocos.csv" "Modelo_com_Precedencias.csv" "modelo_final.csv" 
  python3 genetic_algorithmCPIT.py --instancia modelo_final.csv --pop 80 --geracoes 100 --mutacao 0.08 --seed 123
