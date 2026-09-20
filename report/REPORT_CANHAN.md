# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thành Nam
**Nhóm:** G63
**Ngày:** 2026-09-20
**Chiến lược cá nhân:** Semantic Similarity Chunking

## 1. Khởi động

### Độ tương tự cosine

Cosine similarity đo góc giữa hai vector embedding:

`cos(a,b) = (a · b) / (||a|| × ||b||)`

Giá trị gần 1 nghĩa là hai vector có hướng gần nhau; giá trị gần 0 là ít
tương đồng; giá trị âm là hướng ngược nhau. Hàm `compute_similarity()` trả
`0.0` khi một vector có độ dài bằng 0 để tránh chia cho 0.

### Tính toán chunking

Với 10.000 ký tự, `chunk_size=500`, `overlap=50`, bước dịch là 450:
`ceil((10000 - 50) / 450) = 23` chunks. Khi overlap tăng lên 100, bước dịch
còn 400 và số chunk tăng thành `ceil((10000 - 100) / 400) = 25`. Overlap
giúp giữ ngữ cảnh ở ranh giới nhưng làm tăng chi phí embedding.

## 2. Hướng tiếp cận

Tôi cài đặt `SemanticSimilarityChunker` trong
[`src/chunking.py`](../src/chunking.py). Chunker tách văn bản thành các câu,
embedding từng câu, rồi so sánh cosine similarity của hai câu liên tiếp.
Hai câu được gom vào cùng chunk khi similarity không thấp hơn
`similarity_threshold=0.35` và tổng độ dài không vượt
`max_chunk_size=420`. Khi similarity thấp, chunk mới bắt đầu tại câu đó.

Chiến lược này phù hợp với chính sách đổi trả/bảo hành vì một điều khoản có
thể dài ngắn khác nhau: các câu giải thích cùng một điều kiện được giữ chung,
trong khi chuyển từ điều kiện sang quy trình hoặc ngoại lệ tạo ranh giới tự
nhiên. Đây là strategy riêng, không thay đổi ba key chuẩn của
`ChunkingStrategyComparator`, nên nhóm có thể so sánh công bằng với
Fixed-size, Sentence-based và Recursive.

Embedding function được truyền vào constructor để dùng được model thật,
không khóa chặt vào một nhà cung cấp. `semantic_benchmark.py` ưu tiên
`LocalEmbedder` với model multilingual; nếu máy chưa cài dependency thì nó
dùng mock fallback và in cảnh báo rõ ràng.

## 3. Hoàn thiện code

Đã chạy:

```text
python -m pytest tests/ -q
42 passed in 0.08s
```

Các phần hoàn thiện gồm `SentenceChunker`, `RecursiveChunker`,
`compute_similarity`, `EmbeddingStore`, `KnowledgeBaseAgent` và strategy riêng
`SemanticSimilarityChunker`. Không sửa file kiểm thử.

Artifact chạy strategy riêng:

```text
python bench.py
```

Output đầy đủ được lưu tại
[`SEMANTIC_SIMILARITY_RESULTS.md`](./SEMANTIC_SIMILARITY_RESULTS.md) để
nhóm trưởng đưa vào Phase 2.

## 4. So sánh baseline và strategy riêng

Các con số dưới đây chạy trên ba tài liệu cùng corpus, sau khi bỏ YAML
frontmatter. Ba baseline dùng `chunk_size=420`; Semantic Similarity dùng
`threshold=0.35`, `max_chunk_size=420`.

| Tài liệu | Strategy | Số chunk | Độ dài trung bình | Nhận xét coherence |
|---|---|---:|---:|---|
| buyer-return-request | Fixed-size | 2 | 289.0 | Có thể cắt giữa ý |
| buyer-return-request | Sentence-based | 2 | 263.0 | Giữ ranh giới câu |
| buyer-return-request | Recursive | 2 | 263.0 | Ưu tiên đoạn/câu |
| buyer-return-request | Semantic Similarity | 5 | 104.6 | Ranh giới theo similarity, nhưng quá phân mảnh với mock |
| buyer-refund-processing | Fixed-size | 2 | 264.0 | Kích thước ổn định |
| buyer-refund-processing | Sentence-based | 2 | 238.0 | Mạch lạc theo câu |
| buyer-refund-processing | Recursive | 2 | 238.0 | Giữ cấu trúc đoạn |
| buyer-refund-processing | Semantic Similarity | 4 | 118.5 | Tách được các chuyển ý |
| seller-return-handling | Fixed-size | 2 | 279.0 | Có nguy cơ cắt điều khoản |
| seller-return-handling | Sentence-based | 2 | 253.0 | Dễ đọc |
| seller-return-handling | Recursive | 2 | 253.0 | Baseline cân bằng |
| seller-return-handling | Semantic Similarity | 3 | 168.3 | Ít phân mảnh hơn hai tài liệu buyer |

Trong môi trường hiện tại runner dùng `mock fallback (not semantic)` vì chưa có
`sentence-transformers`. Vì vậy số chunk ở trên chỉ xác nhận pipeline và
ranh giới thuật toán, chưa phải bằng chứng cuối cùng về semantic quality.
Khi nhóm trưởng chạy với model multilingual thật, cần thay lại bảng bằng output
model thật.

## 5. Dự đoán độ tương tự cosine

Các cặp sau được chạy với `MockEmbedder`, nên kết quả chỉ kiểm tra tính ổn định
của pipeline chứ không đại diện cho hiểu biết ngữ nghĩa.

| # | Cặp câu | Dự đoán | Điểm thực tế | Kết luận |
|---:|---|---|---:|---|
| 1 | `return refund` — `refund return` | Cao | 0.105583 | Mock không phản ánh tốt từ đồng nghĩa |
| 2 | `seller shipping label` — `buyer refund timing` | Thấp | -0.173691 | Phù hợp dự đoán |
| 3 | `warranty defect` — `warranty claim` | Cao | -0.100177 | Mock thất bại với cùng chủ đề |
| 4 | `return request deadline` — `python loop` | Thấp | 0.306899 | Mock tạo false positive |
| 5 | `refund approved` — `refund released` | Cao | -0.097398 | Mock không hiểu paraphrase |

Sai lệch này là lý do không dùng mock để tuyên bố strategy semantic thắng.
Model multilingual thật là điều kiện cần cho thí nghiệm chính thức.

## 6. Kết quả retrieval của strategy riêng

Runner dùng cùng 5 query và cùng metadata filter với nhóm. Với mock fallback,
top-3 được ghi đầy đủ trong
[`SEMANTIC_SIMILARITY_RESULTS.md`](./SEMANTIC_SIMILARITY_RESULTS.md).

| # | Query | Filter | Top-1 | Đánh giá nội dung |
|---:|---|---|---|---|
| 1 | Buyer return window | `buyer` | `buyer-return-request#2` | Liên quan |
| 2 | Seller response duties | `seller` | `seller-return-handling#1` | Liên quan |
| 3 | Refund timing | `buyer` | `buyer-received-wrong-item#1` | Một phần |
| 4 | Eligible reasons | Không | `buyer-return-request#4` | Liên quan |
| 5 | Seller shipping | `seller` | `seller-return-handling#1` | Liên quan |

Metadata filter giúp loại các chunk sai audience trước khi xếp hạng, nên vẫn
cần giữ nguyên trong so sánh nhóm. Không được so sánh một strategy có filter
với strategy khác không filter.

## 7. Failure analysis và bài học

**Failure case:** câu 3 về thời điểm hoàn tiền có top-1 là tài liệu
`buyer-received-wrong-item`, còn chunk `buyer-refund-processing` chỉ ở top-3.
Nguyên nhân chính là mock embedding không phân biệt tốt các câu cùng chủ đề
“hoàn tiền”; ngoài ra semantic chunking tạo nhiều chunk ngắn nên thông tin
thời điểm bị chia thành nhiều ứng viên.

**Cải thiện:** chạy lại bằng
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, thử threshold
0.25/0.35/0.45 trên cùng corpus, rồi chọn theo top-3 có chứa đúng câu trả lời.
Nếu chunk vẫn quá ngắn, tăng threshold hoặc thêm minimum chunk size trước khi
cho phép cắt. Không thay đổi corpus hoặc query giữa các strategy.

**Bài học:** chunk boundary ảnh hưởng trực tiếp đến grounding; top-3 cùng
`doc_id` chưa đủ, chunk phải chứa đúng bằng chứng trả lời. Semantic strategy
có tiềm năng xử lý tài liệu chính sách không đều độ dài, nhưng phụ thuộc mạnh
vào embedding model. Với dữ liệu ngắn và có cấu trúc rõ, Recursive/Sentence
có thể ổn định hơn và rẻ hơn.

## 8. Artifact bàn giao cho nhóm trưởng

1. [`src/chunking.py`](../src/chunking.py): implementation và API của
   `SemanticSimilarityChunker`.
2. [`src/__init__.py`](../src/__init__.py): export strategy để import trực tiếp.
3. [`semantic_benchmark.py`](../semantic_benchmark.py): runner dùng đúng corpus,
   5 query, metadata filter và format top-3.
4. [`SEMANTIC_SIMILARITY_RESULTS.md`](./SEMANTIC_SIMILARITY_RESULTS.md):
   output chạy thực tế trong môi trường hiện tại.

Lệnh tích hợp cho nhóm:

```python
from src.chunking import SemanticSimilarityChunker
from src.embeddings import LocalEmbedder

chunker = SemanticSimilarityChunker(
    embedding_fn=LocalEmbedder(),
    similarity_threshold=0.35,
    max_chunk_size=420,
)
chunks = chunker.chunk(content)
```

Khi tổng hợp `REPORT_NHOM.md`, nhóm trưởng nên giữ bốn cột chung:
`count`, `avg_length`, top-3 retrieval và ghi chú coherence; chạy mọi strategy
trên cùng corpus, cùng embedding backend, cùng 5 query và cùng filter.

## Tự đánh giá

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Khởi động | 5 / 5 |
| Hướng tiếp cận | 10 / 10 |
| Hoàn thiện code — tests | 30 / 30 |
| Dự đoán độ tương tự | 5 / 5 |
| Kết quả truy xuất | 10 / 10 |
| **Tổng** | **60 / 60** |
