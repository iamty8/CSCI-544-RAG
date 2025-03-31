# 🔍 Efficient Retrieval-Augmented Generation (RAG) Pipeline

This repository contains the implementation for our project on **RAG with Efficient Retrieval**, focusing on optimizing retrieval efficiency and query reformulation strategies for faster and more accurate generation in large language models.

## 📚 Overview

Large Language Models (LLMs) like ChatGPT, LLaMA, and DeepSeek have shown great capabilities in NLP tasks but suffer from static knowledge and hallucination. **Retrieval-Augmented Generation (RAG)** solves this by enabling LLMs to fetch information from external databases in real-time.

This project explores and compares multiple retrieval techniques and query reformulation strategies to:
- Reduce computational overhead
- Improve retrieval quality
- Accelerate response time

## 📦 Features

- ✅ Modular retriever interface for BM25, FAISS, ANN (HNSW, PQ)
- ✅ Query reformulation strategies: truncation, paraphrasing, and embedding refinement
- ✅ Metrics: Retrieval Speed, MRR@10, Recall@K, ROUGE, BERTScore
- ✅ Built on top of [LlamaIndex](https://github.com/jerryjliu/llama_index) for ease of RAG integration
- ✅ Evaluation with [MS MARCO](https://microsoft.github.io/msmarco/) dataset

## 📊 Evaluation Metrics

We use both retrieval-based and generation-based metrics:
- **Retrieval Speed**
- **Memory Usage**
- **MRR@10**
- **Recall@K**
- **ROUGE / BERTScore**
- **Retrieval Precision**

Optional: human evaluation for coherence and relevance.

## 🧪 Experimental Setup

- **Retriever Backend**: BM25, FAISS, HNSW, PQ  
- **Embedding Model**: Sentence Transformers / Hugging Face LLM  
- **LLM for Paraphrasing**: LLaMA / GPT via API or local models  
- **Evaluation Dataset**: MS MARCO

## 📅 Project Timeline

| Phase             | Dates            | Goals                                                             |
|------------------|------------------|-------------------------------------------------------------------|
| Preparation       | Mar 7 - Mar 20   | Literature review, dataset prep, baseline setup                  |
| Experimentation   | Mar 21 - Apr 3   | Run retrieval & reformulation experiments                        |
| Finalization      | Apr 3 - Apr 24   | Benchmarking, analysis, report writing                           |

## 📖 References

- Lewis et al. (2020) – [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)  
- Zhu et al. (2025) – Sparse RAG  
- Jin et al. (2024) – RAGCache  
- Liu et al. (2024) – FlashBack  
- Johnson et al. (2019) – FAISS


## 🔧 Setup Instructions

1. Create virtual environment:
   ```bash
   cd CSCI-544-RAG
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download MS MARCO dataset:
   ```bash
   python data/download_dataset.py
   ```

4. Run Pipeline & Framework Test Scripts:

   - **Test FAISSRetriever**  
     (Encodes each document into vector embeddings using a SentenceTransformer model):
     ```bash
     python pipeline/test_faiss_retriever.py
     ```

   - **Test BM25Retriever**  
     (Represents each document using sparse TF-IDF-style term frequencies):  
     📝 _Note: Line 41 in `test_bm25_retriever.py` limits the corpus to 10,000 passages for faster indexing._
     ```bash
     python pipeline/test_bm25_retriever.py
     ```

   - **Test ANNRetriever**  
     (Uses approximate nearest neighbor search over dense embeddings to retrieve semantically similar documents):
     ```bash
     python pipeline/test_ann_retriever.py
     ```