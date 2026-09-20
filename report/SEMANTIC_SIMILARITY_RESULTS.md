# Semantic Similarity Chunking — Handoff Artifact

Generated with:

```text
python semantic_benchmark.py
```

Environment result: `mock fallback (not semantic)`. Install the optional
multilingual `sentence-transformers` backend and rerun before using scores as
semantic-quality evidence.

## Configuration

- `similarity_threshold=0.35`
- `max_chunk_size=420`
- Corpus: 7 Shopee return/refund documents
- Indexed chunks: 29

## Baseline statistics

| Document | Fixed | Sentence | Recursive | Semantic |
|---|---:|---:|---:|---:|
| buyer-return-request | 2 / 289.0 | 2 / 263.0 | 2 / 263.0 | 5 / 104.6 |
| buyer-refund-processing | 2 / 264.0 | 2 / 238.0 | 2 / 238.0 | 4 / 118.5 |
| seller-return-handling | 2 / 279.0 | 2 / 253.0 | 2 / 253.0 | 3 / 168.3 |

Each cell is `chunk_count / average_characters`.

## Top-3 retrieval

| # | Query | Filter | Top 1 | Top 2 | Top 3 |
|---:|---|---|---|---|---|
| 1 | Buyer return window | buyer | buyer-return-request#2 (0.353310) | buyer-return-request#3 (0.319064) | buyer-received-wrong-item#2 (0.273727) |
| 2 | Seller response duties | seller | seller-return-handling#1 (0.217181) | seller-return-handling#2 (0.083965) | seller-warranty-obligations#3 (0.072105) |
| 3 | Refund timing | buyer | buyer-received-wrong-item#1 (0.218424) | buyer-return-request#2 (0.150901) | buyer-refund-processing#0 (0.127556) |
| 4 | Eligible reasons | none | buyer-return-request#4 (0.181924) | buyer-received-wrong-item#2 (0.162114) | seller-refund-dispute#1 (0.151293) |
| 5 | Seller shipping | seller | seller-return-handling#1 (0.274082) | seller-refund-dispute#0 (0.227502) | seller-return-handling#2 (0.017160) |

Scores are produced by the selected embedding backend. For the current
fallback, evaluate relevance by checking chunk content against the gold
answer, not by score alone.
