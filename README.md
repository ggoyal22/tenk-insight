---
title: Tenk Insight
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Tenk Insight

A retrieval-augmented generation (RAG) system for querying SEC 10-K filings using natural language.

## What it does

Ask questions about publicly filed 10-K annual reports and get answers grounded in the source documents. The system retrieves relevant passages, ranks them, and synthesises a response with citations.

## Stack

- **Retrieval** — hybrid vector + keyword search with RRF fusion and cross-encoder reranking
- **Generation** — LangGraph pipeline with multi-hop retrieval and self-reflection
- **Embeddings** — `BAAI/bge-large-en-v1.5`
- **UI** — Streamlit

## Setup

Copy `.env.example` to `.env` and fill in the required values, then run:

```bash
docker compose up
```
