# Dự án Constitutional AI với RLAIF và GPT-2

## 📖 Giới thiệu

Dự án này là một triển khai nâng cao của phương pháp **Constitutional AI (CAI)**, kết hợp với **Reinforcement Learning from AI Feedback (RLAIF)** để huấn luyện một mô hình ngôn ngữ (GPT-2) trở nên hữu ích, trung thực, vô hại và tôn trọng hơn.

Điểm đặc biệt của dự án là sử dụng một mô hình "nhà phê bình" (AI Critic) để tự động đánh giá và cung cấp tín hiệu thưởng, loại bỏ nhu cầu về dữ liệu sở thích do con người gán nhãn, giúp quá trình tinh chỉnh mô hình trở nên tự động và có khả năng mở rộng.

Toàn bộ mã nguồn được cung cấp trong một tệp Python duy nhất, bao gồm tất cả các logic từ tạo dữ liệu, huấn luyện, đánh giá, đến trực quan hóa và tương tác.

### ✨ Tính năng chính của Mã nguồn

-   **Huấn luyện Constitutional AI**: Tinh chỉnh GPT-2 dựa trên các nguyên tắc đã được định nghĩa.
-   **Cơ chế RLAIF**: Sử dụng mô hình thưởng (Reward Model) để học từ các đánh giá của AI Critic.
-   **Tạo dữ liệu đa dạng**: Tự động tạo ra một tập dữ liệu lớn để huấn luyện.
-   **Đánh giá toàn diện**: So sánh hiệu suất của mô hình trước và sau khi huấn luyện.
-   **Trực quan hóa nâng cao**: Tạo biểu đồ và dashboard tương tác để theo dõi và phân tích kết quả.
-   **Lưu và tải mô hình**: Lưu lại mô hình đã huấn luyện và các kết quả liên quan.
-   **Demo tương tác**: Cung cấp giao diện dòng lệnh để người dùng trực tiếp thử nghiệm mô hình.

## ⚙️ Yêu cầu hệ thống

-   Python 3.8+
-   `pip` để quản lý các gói thư viện
-   **GPU (Rất khuyến khích)**: Dự án này bao gồm việc huấn luyện mô hình ngôn ngữ lớn. Việc chạy trên GPU (với CUDA) sẽ nhanh hơn đáng kể. Nếu không có GPU, quá trình huấn luyện sẽ rất chậm.

## 🚀 Hướng dẫn Cài đặt

Thực hiện các bước sau để thiết lập môi trường cho dự án.

### 1. Lưu mã nguồn

Trước tiên, sao chép toàn bộ mã nguồn Python bạn đã có và lưu vào một tệp, ví dụ: `constitutional_ai_demo.py`.

### 2. Tạo và Kích hoạt Môi trường ảo

Sử dụng môi trường ảo là một thực hành tốt để tránh xung đột thư viện.

```bash
# Tạo một môi trường ảo có tên là "venv"
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate```

### 3. Cài đặt các thư viện cần thiết

Tạo một tệp có tên `requirements.txt` trong cùng thư mục dự án và dán nội dung sau vào:

```txt
transformers
torch
datasets
accelerate
peft
trl
wandb
numpy
matplotlib
seaborn
plotly
pandas
Use code with caution.
Markdown
Sau đó, chạy lệnh sau từ cửa sổ terminal đã kích hoạt môi trường ảo để cài đặt tất cả các thư viện:
Generated bash
pip install -r requirements.txt
Use code with caution.
Bash
Lưu ý: Nếu bạn có GPU NVIDIA, hãy đảm bảo rằng bạn đã cài đặt phiên bản torch tương thích với CUDA để tận dụng tối đa hiệu năng. Bạn có thể tham khảo trang chủ của PyTorch để biết lệnh cài đặt chính xác.
▶️ Cách chạy dự án
Mã nguồn được thiết kế để chạy ở các chế độ khác nhau. Bạn cần chỉnh sửa khối if __name__ == "__main__": ở cuối tệp constitutional_ai_demo.py để chọn chế độ mong muốn.
Chế độ 1: Kiểm tra nhanh (Quick Test)
Chế độ này rất hữu ích để kiểm tra xem môi trường của bạn đã được cài đặt đúng cách chưa. Nó sẽ không huấn luyện mô hình mà chỉ kiểm tra việc tải thư viện, tạo văn bản và tính điểm.
Mở tệp constitutional_ai_demo.py.
Tìm đến khối if __name__ == "__main__":.
Bỏ ghi chú (uncomment) dòng quick_test() và đảm bảo các dòng khác được ghi chú (comment out).
Generated python
if __name__ == "__main__":
    # Để kiểm tra nhanh, bỏ ghi chú dòng dưới đây:
    quick_test()

    # Ghi chú các dòng khác để chúng không chạy:
    # run_complete_demo()
    # demo_constitutional_ai_complete()
Use code with caution.
Python
Chạy script từ terminal:
Generated bash
python constitutional_ai_demo.py
Use code with caution.
Bash
Chế độ 2: Chạy Toàn bộ Demo (Full Demo)
Đây là chế độ chính của dự án. Nó sẽ thực hiện toàn bộ quy trình:
Tạo dữ liệu huấn luyện.
Đánh giá mô hình trước khi huấn luyện.
Thực hiện quá trình huấn luyện (có thể mất nhiều thời gian).
Đánh giá mô hình sau khi huấn luyện.
Tạo các biểu đồ trực quan hóa và bảng phân tích.
Lưu lại mô hình và tất cả kết quả.
Hỏi bạn có muốn bắt đầu chế độ demo tương tác không.
⚠️ Cảnh báo: Quá trình này đòi hỏi tài nguyên tính toán đáng kể (đặc biệt là VRAM của GPU) và có thể mất từ 30 phút đến vài giờ tùy thuộc vào cấu hình phần cứng của bạn.
Mở tệp constitutional_ai_demo.py.
Tìm đến khối if __name__ == "__main__":.
Đảm bảo dòng run_complete_demo() được bỏ ghi chú.
Generated python
if __name__ == "__main__":
    # Ghi chú dòng kiểm tra nhanh:
    # quick_test()

    # Bỏ ghi chú dòng dưới đây để chạy toàn bộ demo:
    run_complete_demo()
Use code with caution.
Python
Chạy script từ terminal:
Generated bash
python constitutional_ai_demo.py
Use code with caution.
Bash
Sau khi quá trình huấn luyện và phân tích kết thúc, chương trình sẽ hỏi bạn:
Would you like to try the interactive demo? (y/n):
Nhập y và nhấn Enter để bắt đầu trò chuyện và so sánh trực tiếp các câu trả lời của mô hình.
