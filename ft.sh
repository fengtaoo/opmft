#!/bin/sh
cuda="CUDA_VISIBLE_DEVICES=0,2,3"



cmd="srun -G 4 --mem 80g  python train.py  \
-ranking_loss \
-mse \
-checkpoint /home/fengtao/Projects/ft_openmatch/checkpoints/ft3ever_step64000.bin \
-task new  \
-model bert   \
-train /home/fengtao/Projects/dataset/or_quac_data_v2/om_train_orquac.shuf.jsonl  \
-max_input 12800000  -epoch 1  \
-save checkpoints  \
-dev  /data/private/fengtao/Projects/data/manual_orquac_ance_cdr_with_ranking.jsonl \
-dev2 /home/fengtao/Projects/dataset/or_quac_data_v2/manual_orquac_ance.jsonl  \
-qrels /home/fengtao/Projects/dataset/qrels.test.tsv  \
-qrels2 /home/fengtao/Projects/dataset/qrels.test.tsv  \
-vocab bert-base-uncased  \
-pretrain bert-base-uncased  \
-metric mrr_cut_5  \
-max_query_len 125  \
-max_doc_len 384   \
-batch_size 16  \
-lr 1e-6  \
-eval_every 1000  \
-n_warmup_steps 5000  \
--log_dir=logs/event_logs   \
--teach  "




echo $cmd
eval $cmd
