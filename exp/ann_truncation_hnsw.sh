python ./exp/run.py \
  --retriever "ann" \
  --retriever_args "method=hnsw" \
  --query "How long can I wait to cook my frozen ground turkey which I put in my fridge?" \
  --rewrite_method truncation \
  --top_k 10 