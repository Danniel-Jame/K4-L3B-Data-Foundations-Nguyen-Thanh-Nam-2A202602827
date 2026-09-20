# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thành Nam  
**Nhóm:** G63 
**Ngày:** 2026-09-19

## 1. Khởi động (Warm-up)

### Độ tương tự cosine

Cosine similarity cao nghĩa là hai vector có hướng gần nhau, nên nội dung được
embedding biểu diễn là tương tự về ngữ nghĩa. Ví dụ cao: “yêu cầu hoàn tiền”
và “đề nghị refund”. Ví dụ thấp: “thời hạn đổi trả” và “vòng lặp Python”,
vì chúng thuộc hai chủ đề khác nhau. Cosine phù hợp với text hơn Euclid vì
nó tập trung vào hướng/ngữ nghĩa và ít bị ảnh hưởng bởi độ dài vector.

### Tính toán chunking

Với 10.000 ký tự, `chunk_size=500`, `overlap=50`, bước dịch là 450:
`ceil((10000 - 50) / 450) = 23` chunks. Khi overlap tăng lên 100, bước
dịch còn 400 và số chunk tăng thành `ceil((10000 - 100) / 400) = 25`.
Overlap lớn giúp giữ ngữ cảnh ở ranh giới chunk nhưng làm tăng số chunk và
chi phí embedding.

## 2. Hướng tiếp cận

`SentenceChunker` dùng regex nhận diện dấu `.`, `!`, `?` đi trước khoảng trắng
hoặc xuống dòng, giữ dấu câu rồi gom theo số câu tối đa. Chuỗi rỗng trả về
danh sách rỗng và khoảng trắng thừa được loại bỏ.

`RecursiveChunker` thử các separator theo thứ tự `\n\n`, `\n`, `. `, khoảng
trắng và cuối cùng là ký tự. Đoạn đã đủ ngắn là base case; đoạn dài được tách
thành các đơn vị, gom lại không vượt `chunk_size`, rồi đệ quy với separator
thấp hơn khi cần.

`EmbeddingStore` lưu bản ghi trong bộ nhớ gồm id, content, metadata và vector.
Search embed query, tính dot product trên vector đã chuẩn hóa của mock embedder
và sắp xếp giảm dần. `search_with_filter` lọc metadata trước khi tính điểm;
`delete_document` xóa mọi chunk có `metadata["doc_id"]` bằng id tài liệu gốc.

`KnowledgeBaseAgent.answer` thực hiện retrieval, tạo context có số thứ tự,
source/source_url, metadata và content, rồi đưa context cùng câu hỏi vào
prompt cho `llm_fn`. Vì source được chèn trực tiếp, câu trả lời có thể truy
vết về chunk đã dùng.

## 3. Hoàn thiện code

Đã chạy `python -m pytest tests/ -v`:

```text
42 passed in 0.06s
```

Không sửa file kiểm thử. Các TODO trong `src/chunking.py`, `src/store.py` và
`src/agent.py` đã được triển khai; vector store hoàn toàn chạy trong bộ nhớ và
không dùng ChromaDB.

## 4. Dự đoán độ tương tự

Các điểm dưới đây là kết quả đo với `MockEmbedder` và
`compute_similarity`; backend mock là deterministic nhưng không đảm bảo phản
ánh ngữ nghĩa tự nhiên như mô hình multilingual thật.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|---|---|---|---|---:|---|
| 1 | return refund | refund return | cao | 0.105583 | Có, tương đối |
| 2 | seller shipping label | buyer refund timing | thấp | -0.173691 | Có |
| 3 | warranty defect | warranty claim | cao | -0.100177 | Không |
| 4 | return request deadline | python loop | thấp | 0.306899 | Không |
| 5 | refund approved | refund released | cao | -0.097398 | Không |

Kết quả bất ngờ ở các cặp 3–5: mock embedding sinh vector băm deterministic,
không hiểu synonym hay chủ đề. Điều này cho thấy điểm cosine chỉ có ý nghĩa
ngữ nghĩa khi backend embedding được huấn luyện cho ngôn ngữ và nhiệm vụ phù
hợp; khi dùng mock, benchmark chủ yếu kiểm tra tính đúng và ổn định của pipeline.

## 5. Kết quả truy xuất

Bộ benchmark gồm 5 câu hỏi trong `bench.py`; ba câu dùng bộ lọc buyer/seller
và một câu không lọc. Kết quả đầy đủ được lưu tại `ket_qua_benchmark.txt`.

| # | Câu hỏi | Top-1 chunk | Score | Liên quan? | Tóm tắt |
|---|---|---|---:|---|---|
| 1 | Buyer return window | buyer-return-request | 0.121713 | Có | Yêu cầu hoàn trả/hoàn tiền trong cửa sổ quy định |
| 2 | Seller response duties | seller-warranty-obligations | 0.198426 | Một phần | Nghĩa vụ người bán và xử lý yêu cầu |
| 3 | Refund timing | buyer-received-wrong-item | 0.205785 | Một phần | Quy trình hoàn tiền sau khi xử lý |
| 4 | Eligible reasons | buyer-return-request | 0.295393 | Có | Lý do và bằng chứng cho khiếu nại |
| 5 | Seller shipping | seller-refund-dispute | 0.251701 | Có | Tranh chấp, vận chuyển và xử lý kiện hàng |

Có chunk liên quan trong top-3 ở **5/5** câu theo kiểm tra thủ công nội dung
corpus. Bộ lọc metadata loại bỏ các tài liệu sai đối tượng trước khi xếp hạng,
đặc biệt hữu ích cho câu 1–3 và 5.

## Tự Đánh Giá

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Khởi động | 5 / 5 |
| Hướng tiếp cận | 10 / 10 |
| Hoàn thiện code — tests | 30 / 30 |
| Dự đoán độ tương tự | 5 / 5 |
| Kết quả truy xuất | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
