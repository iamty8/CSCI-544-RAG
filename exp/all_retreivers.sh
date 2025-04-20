~/.conda/envs/rlbench/envs/csci544_rag/bin/python ./exp/run.py \
  --retriever "ann" \
  --retriever_args "method=hnsw" \
  --top_k 10 

~/.conda/envs/rlbench/envs/csci544_rag/bin/python ./exp/run.py \
  --retriever "dense" \
  --top_k 10 

~/.conda/envs/rlbench/envs/csci544_rag/bin/python ./exp/run.py \
  --retriever "bm25" \
  --retriever_args "fuzzy_threshold=40" \
  --top_k 10 

~/.conda/envs/rlbench/envs/csci544_rag/bin/python ./exp/run.py \
  --retriever "faiss" \
  --top_k 10 