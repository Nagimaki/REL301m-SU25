# Dự án Constitutional AI với RLAIF và GPT-2

Dự án này là một triển khai nâng cao của phương pháp **Constitutional AI (CAI)**, kết hợp với **Reinforcement Learning from AI Feedback (RLAIF)** để huấn luyện một mô hình ngôn ngữ (GPT-2) trở nên hữu ích, trung thực, vô hại và tôn trọng hơn.

Điểm đặc biệt của dự án này là sử dụng một mô hình "nhà phê bình" (AI Critic) để tự động đánh giá và cung cấp tín hiệu thưởng, loại bỏ nhu cầu về dữ liệu sở thích do con người gán nhãn, giúp quá trình tinh chỉnh mô hình trở nên tự động và có khả năng mở rộng.

## ✨ Tính năng chính

-   **Huấn luyện Constitutional AI**: Tinh chỉnh GPT-2 dựa trên các nguyên tắc đã được định nghĩa (Vô hại, Hữu ích, Trung thực, Tôn trọng).
-   **Cơ chế RLAIF**: Sử dụng mô hình thưởng (Reward Model) để học từ các đánh giá của AI Critic, tạo ra tín hiệu loss cho việc huấn luyện.
-   **Tạo dữ liệu đa dạng**: Tự động tạo ra một tập dữ liệu lớn và đa dạng để huấn luyện, giúp mô hình khái quát hóa tốt hơn.
-   **Đánh giá toàn diện**: So sánh hiệu suất của mô hình trước và sau khi huấn luyện dựa trên nhiều chỉ số.
-   **Trực quan hóa nâng cao**: Tạo biểu đồ và dashboard tương tác để theo dõi tiến trình huấn luyện và phân tích kết quả.
-   **Lưu và tải mô hình**: Lưu lại mô hình đã huấn luyện, các chỉ số và kết quả để sử dụng trong tương lai.
-   **Demo tương tác**: Cung cấp một giao diện dòng lệnh để người dùng có thể trực tiếp tương tác và so sánh mô hình gốc với mô hình đã được tinh chỉnh.

## 📂 Cấu trúc dự án

Toàn bộ mã nguồn được chứa trong một tệp duy nhất để dễ dàng trình bày. Khi bạn chạy dự án, các kết quả sau sẽ được tạo ra:
/
|-- your_script_name.py # Tệp mã nguồn chính của bạn
|-- README.md # Tệp hướng dẫn này
|-- requirements.txt # Tệp chứa các thư viện cần thiết
|
|-- constitutional_gpt2_enhanced/ # Thư mục chứa mô hình đã được huấn luyện
| |-- config.json
| |-- generation_config.json
| |-- model.safetensors
| |-- tokenizer_config.json
| |-- vocab.json
| |-- merges.txt
| |-- reward_model.pt # Mô hình thưởng đã huấn luyện
| |-- training_metrics.json # Dữ liệu về quá trình huấn luyện
| -- evaluation_results.json # Kết quả đánh giá | |-- training_progress.png # Biểu đồ tiến trình huấn luyện-- principle_comparison.png # Biểu đồ so sánh các nguyên tắc


## ⚙️ Yêu cầu hệ thống

-   Python 3.8+
-   `pip` để quản lý các gói thư viện
-   **GPU (Rất khuyến khích)**: Dự án này bao gồm việc huấn luyện mô hình ngôn ngữ lớn. Việc chạy trên GPU (với CUDA) sẽ nhanh hơn đáng kể. Nếu không có GPU, quá trình huấn luyện sẽ rất chậm.

## 🚀 Hướng dẫn Cài đặt

Thực hiện các bước sau để thiết lập môi trường cho dự án.

### 1. Sao chép (Clone) Dự án

```bash
# Nếu bạn có một kho git, nếu không, chỉ cần tạo một thư mục và đặt tệp mã nguồn vào đó
git clone <your-repository-url>
cd <your-project-directory>

# Tạo một môi trường ảo có tên là "venv"
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate```

### 3. Cài đặt các thư viện cần thiết

Tạo một tệp có tên `requirements.txt` và dán nội dung sau vào:

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

pip install -r requirements.txt```

**Lưu ý**: Nếu bạn có GPU NVIDIA, hãy đảm bảo rằng bạn đã cài đặt phiên bản `torch` tương thích với CUDA để tận dụng tối đa hiệu năng. Bạn có thể tham khảo trang chủ của [PyTorch](https://pytorch.org/get-started/locally/) để biết lệnh cài đặt chính xác.

## ▶️ Cách chạy dự án

Mã nguồn được thiết kế để chạy ở các chế độ khác nhau. Bạn cần chỉnh sửa dòng cuối cùng của tệp mã nguồn để chọn chế độ mong muốn.

### Chế độ 1: Kiểm tra nhanh (Quick Test)

Chế độ này rất hữu ích để kiểm tra xem môi trường của bạn đã được cài đặt đúng cách chưa. Nó sẽ không huấn luyện mô hình mà chỉ kiểm tra việc tải mô hình, tạo văn bản và tính điểm.

1.  Mở tệp mã nguồn Python.
2.  Tìm đến khối `if __name__ == "__main__":`.
3.  Bỏ ghi chú (uncomment) dòng `quick_test()`.

    ```python
    if __name__ == "__main__":
        # Bỏ ghi chú dòng dưới đây để kiểm tra nhanh
        quick_test()
    
        # Ghi chú các dòng khác
        # run_complete_demo()
    ```

4.  Chạy script từ terminal:

    ```bash
    python your_script_name.py
    ```

### Chế độ 2: Chạy Toàn bộ Demo (Full Demo)

Đây là chế độ chính của dự án. Nó sẽ thực hiện toàn bộ quy trình:
1.  Tạo dữ liệu huấn luyện.
2.  Đánh giá mô hình trước khi huấn luyện.
3.  **Thực hiện quá trình huấn luyện (có thể mất nhiều thời gian)**.
4.  Đánh giá mô hình sau khi huấn luyện.
5.  Tạo các biểu đồ trực quan hóa và bảng phân tích.
6.  Lưu lại mô hình và tất cả kết quả.
7.  Hỏi bạn có muốn bắt đầu chế độ demo tương tác không.

**⚠️ Cảnh báo**: Quá trình này đòi hỏi tài nguyên tính toán đáng kể (đặc biệt là VRAM của GPU) và có thể mất từ 30 phút đến vài giờ tùy thuộc vào cấu hình phần cứng của bạn.

1.  Mở tệp mã nguồn Python.
2.  Tìm đến khối `if __name__ == "__main__":`.
3.  Đảm bảo dòng `run_complete_demo()` được bỏ ghi chú.

    ```python
    if __name__ == "__main__":
        # Ghi chú dòng kiểm tra nhanh
        # quick_test()
    
        # Bỏ ghi chú dòng dưới đây để chạy toàn bộ demo
        run_complete_demo()
    ```

4.  Chạy script từ terminal:

    ```bash
    python your_script_name.py
    ```

5.  Sau khi quá trình huấn luyện và phân tích kết thúc, chương trình sẽ hỏi bạn:
    `Would you like to try the interactive demo? (y/n):`
    Nhập `y` và nhấn Enter để bắt đầu trò chuyện và so sánh trực tiếp các câu trả lời của mô hình.

## 📜 License

Dự án này được cấp phép theo Giấy phép MIT. Xem tệp `LICENSE` để biết thêm chi tiết.
