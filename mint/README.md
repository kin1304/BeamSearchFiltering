# MINT - Text Graph Library

MINT (Text Graph Library) là một thư viện Python để xây dựng và phân tích đồ thị văn bản từ tiếng Việt sử dụng py_vncorenlp.

## 🚀 Tính năng chính

### Xây dựng đồ thị văn bản
- **Word nodes**: Chứa từng từ trong context và claim với thông tin POS tag, lemma
- **Sentence nodes**: Các câu trong context
- **Claim node**: Nội dung claim cần kiểm tra

### Phân tích thông minh
- ✅ Tìm từ chung giữa context và claim
- ✅ Thống kê tần suất từ
- ✅ Phân tích cấu trúc đồ thị
- ✅ Xuất dữ liệu ra JSON
- ✅ Lưu/tải đồ thị

### Visualization
- ✅ Vẽ đồ thị với màu sắc phân biệt các loại node
- ✅ Interactive graph layout

## 📦 Cài đặt

```bash
pip install py_vncorenlp networkx matplotlib numpy
```

## 🔧 Sử dụng cơ bản

```python
from mint import TextGraph
import py_vncorenlp

# Khởi tạo model
model = py_vncorenlp.VnCoreNLP(save_dir="vncorenlp")

# Dữ liệu
context = "Văn bản context..."
claim = "Văn bản claim..."

# Xử lý với py_vncorenlp
context_sentences = model.annotate_text(context)
claim_sentences = model.annotate_text(claim)

# Tạo đồ thị
text_graph = TextGraph()
text_graph.build_from_vncorenlp_output(context_sentences, claim, claim_sentences)

# Thống kê
stats = text_graph.get_detailed_statistics()
print(f"Tổng nodes: {stats['total_nodes']}")
print(f"Từ chung: {stats['shared_words_count']}")

# Vẽ đồ thị
text_graph.visualize()
```

## 📊 Các phương thức chính

### Xây dựng đồ thị
- `build_from_vncorenlp_output()`: Xây dựng đồ thị từ output của py_vncorenlp
- `add_word_node()`: Thêm word node
- `add_sentence_node()`: Thêm sentence node
- `add_claim_node()`: Thêm claim node

### Phân tích
- `get_statistics()`: Thống kê cơ bản
- `get_detailed_statistics()`: Thống kê chi tiết
- `get_shared_words()`: Tìm từ chung
- `get_word_frequency()`: Thống kê tần suất từ

### I/O
- `save_graph()`: Lưu đồ thị ra file GEXF
- `load_graph()`: Tải đồ thị từ file
- `export_to_json()`: Xuất ra JSON

### Visualization
- `visualize()`: Vẽ đồ thị

## 🎯 Ứng dụng cho Fact-checking

Thư viện này được thiết kế đặc biệt cho các ứng dụng fact-checking:

1. **Semantic Similarity**: So sánh độ tương đồng giữa claim và context
2. **Evidence Detection**: Tìm evidence supporting/contradicting
3. **Linguistic Analysis**: Phân tích cấu trúc ngôn ngữ
4. **Feature Extraction**: Trích xuất features cho ML models

## 📈 Mở rộng

Thư viện được thiết kế modular, dễ dàng mở rộng:

- Thêm các loại node mới (Entity, Relation, etc.)
- Tích hợp thêm NLP tools
- Xây dựng các metric similarity tùy chỉnh
- Hỗ trợ thêm định dạng export/import

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Hãy tạo issue hoặc pull request.

## 📄 License

MIT License 