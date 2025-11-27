Para rodar o TopoSort:

python generate_precedence.py --arq_blocos kd_block_model.csv --arq_saida kd_precedence.csv

python baseline_toposort.py --blocks kd_block_model.csv --precedence kd_precedence.csv

Para rodar o GRASP:

python grasp.py --arq_blocos kd_block_model.csv --arq_precs kd_precedence.csv \
    --iterations 50 --alpha 0.25
