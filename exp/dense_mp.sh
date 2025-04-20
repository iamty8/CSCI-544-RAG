nohup ~/.conda/envs/rlbench/envs/csci544_rag/bin/python ./exp/run.py \
  --retriever "dense" \
  --query_method truncation \
  --top_k 10 \
  --rank 0 > nohup_rank0.out 2>&1 &

nohup ~/.conda/envs/rlbench/envs/csci544_rag/bin/python ./exp/run.py \
  --retriever "dense" \
  --query_method refine \
  --top_k 10 \
  --rank 1 > nohup_rank1.out 2>&1 &

nohup ~/.conda/envs/rlbench/envs/csci544_rag/bin/python ./exp/run.py \
  --retriever "dense" \
  --query_method paraphrase \
  --top_k 10 \
  --rank 2 > nohup_rank2.out 2>&1 &

nohup ~/.conda/envs/rlbench/envs/csci544_rag/bin/python ./exp/run.py \
  --retriever "dense" \
  --query_method no-rewrite \
  --top_k 10 \
  --rank 3 > nohup_rank3.out 2>&1 &