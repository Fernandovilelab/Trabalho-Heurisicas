Para gerar o modelo de precedencia (restrição geografica)
python generate_precedence.py --arq_blocos Modelo_de_Blocos.csv --arq_saida block_precedence.csv

Para rodar o baseline toposort:

python baseline_toposort.py --blocks kd_block_model.csv --precedence kd_precedence.csv


Arquivos necessários

Modelo de Blocos.csv — modelo de blocos (separador ;)

Modelo_com_Precedencias.csv — relações de precedência (separador ,)

  python3 baseline_toposort.py --arq_blocos Modelo_de_Blocos.csv --arq_precs Modelo_com_Precedencias.csv --arq_out_blocks blocos_ordenados.csv --arq_out_sum resumo.csv --proc_cap_tpy 12000000 --discount 0.12  --anos 12


Para rodar o GRASP:

python grasp.py --arq_blocos kd_block_model.csv --arq_precs kd_precedence.csv \
    --iterations 50 --alpha 0.25
