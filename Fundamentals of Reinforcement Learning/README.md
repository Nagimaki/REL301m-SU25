
# Bài học 1: Vấn đề K-Armed Bandit

## 1. Phần thưởng (Reward)
- **Định nghĩa:** Giá trị nhận được ngay sau khi thực hiện một hành động, phản ánh mức độ “tốt” hay “hiệu quả” của hành động đó.
- **Ký hiệu:** \(R_t\) là phần thưởng tại thời điểm \(t\).
- **Ví dụ bác sĩ:**
  - Bác sĩ có 3 loại thuốc (A, B, C) điều trị một bệnh.
  - Khi kê thuốc A, bệnh nhân hồi phục nhanh → \(R_t = +10\).
  - Nếu bệnh nhân gặp tác dụng phụ nhẹ → \(R_t = -2\).

## 2. Tính chất thời gian của vấn đề Bandit (Temporal Property)
- **Định nghĩa:** Chuỗi hành động và phần thưởng xảy ra tuần tự theo thời gian; lựa chọn hiện tại ảnh hưởng đến kết quả và chiến lược tương lai.
- **Biểu diễn:**
  \[
    A_1, R_1, A_2, R_2, \dots, A_t, R_t
  \]
- **Ví dụ bác sĩ:**
  - Ngày 1 bác sĩ kê thuốc A → bệnh nhân cải thiện nhẹ (\(R_1 = +5\)).
  - Ngày 2, dựa vào kết quả hôm trước, bác sĩ đổi sang thuốc B → bệnh nhân hồi phục tốt hơn (\(R_2 = +8\)).

## 3. K-Armed Bandit
- **Định nghĩa:** Mô hình với \(K\) hành động (arms), mỗi hành động cho phần thưởng ngẫu nhiên theo phân phối khác nhau; mục tiêu tối đa hóa tổng phần thưởng.
- **Mục tiêu:**
  \[
    \max \sum_{t=1}^T R_t
  \]
- **Ví dụ bác sĩ:**
  - Bác sĩ có 4 phương pháp điều trị: thuốc A, thuốc B, phẫu thuật, vật lý trị liệu.
  - Mỗi phương pháp cho phần thưởng khác nhau tùy bệnh nhân (ví dụ: phẫu thuật hồi phục nhanh nhưng rủi ro cao).
  - Bác sĩ phải thử nghiệm và chọn phương pháp tốt nhất cho nhóm bệnh nhân.

## 4. Giá trị hành động (Action Value)
- **Định nghĩa:** Kỳ vọng phần thưởng nếu luôn chọn hành động đó.
- **Giá trị thực:**
  \[
    q_*(a) = \mathbb{E}[R_t \mid A_t = a]
  \]
- **Ước lượng (Sample Average):**
  \[
    Q_n(a) = \frac{r_1 + r_2 + \dots + r_n}{N(a)}
  \]
  - \(N(a)\) là số lần đã chọn hành động \(a\).
- **Ví dụ bác sĩ:**
  - Bác sĩ đã kê thuốc A cho 5 bệnh nhân, nhận phần thưởng \(+6, +7, +5, +8, +6\).
  - \[
      Q(A) = \frac{6 + 7 + 5 + 8 + 6}{5} = 6.4
    \]

---

# Bài học 2: Học gì? Ước lượng giá trị hành động

## 1. Phương pháp ước lượng giá trị hành động
- **Định nghĩa:** Cách tính toán giá trị kỳ vọng của từng hành động dựa trên phần thưởng thu thập từ kinh nghiệm.
- **Công thức Sample Average:**
  \[
    Q_n(a) = \frac{\sum_{i=1}^n r_i}{N(a)}
  \]
- **Ví dụ bác sĩ:**
  - Kê thuốc B cho 3 bệnh nhân, phần thưởng \(+7, +6, +8\).
  - \[Q(B) = \frac{7 + 6 + 8}{3} = 7\]

## 2. Khám phá và khai thác (Exploration vs. Exploitation)
- **Định nghĩa:**
  - **Khai thác:** Chọn hành động có giá trị cao nhất hiện tại để tối đa hóa phần thưởng ngắn hạn.
  - **Khám phá:** Thử hành động mới để thu thập thông tin, tối ưu hóa lâu dài.
- **Ví dụ bác sĩ:** Thường kê thuốc B (khai thác) nhưng thỉnh thoảng thử thuốc C (khám phá) để phát hiện hiệu quả tiềm năng.

## 3. Chọn hành động tham lam theo giá trị hành động (Greedy)
- **Định nghĩa:** Luôn chọn hành động có ước lượng giá trị \(Q(a)\) cao nhất.
- **Công thức:**
  \[
    A_t = \arg\max_a Q(a)
  \]
- **Ví dụ bác sĩ:** Giữa thuốc A (6.4), B (7), C (5) → luôn chọn thuốc B.

## 4. Học trực tuyến (Online Learning)
- **Định nghĩa:** Cập nhật giá trị hành động ngay khi nhận được phần thưởng mới, không phải chờ hết tập dữ liệu.

## 5. Phương pháp trung bình mẫu đơn giản (Sample Average Method)
- **Cập nhật đệ quy:**
  \[
    Q_{n+1} = Q_n + \frac{1}{n}(R_n - Q_n)
  \]
- **Ví dụ bác sĩ:**
  - \(Q_3 = 6.4\), phần thưởng lần 4 là \(R_4 = 9\):
  \[
    Q_4 = 6.4 + \frac{1}{4}(9 - 6.4) = 7.05
  \]

## 6. Phương trình cập nhật trực tuyến tổng quát (General Update)
- **Công thức với learning rate \(\alpha\):**
  \[
    Q_{n+1} = Q_n + \alpha (R_n - Q_n), \quad 0<\alpha\le1
  \]
- **Ví dụ bác sĩ:**
  - \(Q_n = 6.4\), \(R=9\), chọn \(\alpha=0.1\):
  \[
    Q' = 6.4 + 0.1(9 - 6.4) = 6.66
  \]

## 7. Bước cập nhật không đổi cho tình trạng không ổn định
- **Định nghĩa:** Dùng \(\alpha\) cố định để nhanh chóng thích nghi với môi trường thay đổi (non-stationary).
- **Công thức:**
  \[
    Q_{n+1} = Q_n + \alpha (R_n - Q_n)
  \]
- **Ví dụ bác sĩ:** Virus đột biến, hiệu quả thuốc thay đổi → chọn \(\alpha=0.1\) cố định để cập nhật nhanh.

---

# Bài học 3: Sự đánh đổi giữa khám phá và khai thác

## 1. Epsilon-Greedy (ε-Greedy)
- **Định nghĩa:** Với xác suất \(1-\epsilon\) chọn hành động tốt nhất, với xác suất \(\epsilon\) chọn ngẫu nhiên.
- **Công thức:**
  \[
    A_t = \begin{cases}
      \arg\max_a Q(a), & \text{với xác suất }1-\epsilon, \\
      \text{random action}, & \text{với xác suất }\epsilon.
    \end{cases}
  \]
- **Ví dụ bác sĩ:** \(\epsilon=0.1\) → 90% chọn thuốc tốt nhất, 10% thử thuốc khác.

## 2. Lợi ích ngắn hạn của khai thác và dài hạn của khám phá
- **Khai thác:** Tối đa phần thưởng ngay lập tức.
- **Khám phá:** Phát hiện hành động tốt hơn cho tương lai.
- **Ví dụ bác sĩ:** Khai thác thuốc B (7) ngắn hạn, khám phá thuốc C có thể là 9 dài hạn.

## 3. Giá trị khởi tạo lạc quan (Optimistic Initial Values)
- **Định nghĩa:** Gán giá trị ước lượng cao khởi đầu cho mọi hành động để khuyến khích khám phá.
- **Ví dụ bác sĩ:** Ban đầu đặt \(Q(A)=Q(B)=Q(C)=10\) để đảm bảo thử hết cả ba thuốc.

## 4. Lợi ích của giá trị khởi tạo lạc quan
- Đảm bảo mọi hành động được thử ít nhất một lần, tránh bỏ sót lựa chọn tốt.

## 5. Phê bình giá trị khởi tạo lạc quan
- Sau khi đủ khám phá, các giá trị hội tụ, giá trị khởi tạo lạc quan không còn tác động.

## 6. Phương pháp UCB (Upper Confidence Bound)
- **Định nghĩa:** Chọn \(a\) tối đa hóa tổng giá trị ước lượng và mức độ không chắc chắn.
- **Công thức:**
  \[
    A_t = \arg\max_a \Bigl(Q(a) + c \sqrt{\frac{\ln t}{N(a)}}\Bigr)
  \]
  - \(c\) điều chỉnh độ khám phá.
- **Ví dụ bác sĩ:** Thuốc mới thử ít lần có phần bù lớn → được thử thêm.

## 7. Lạc quan trong sự không chắc chắn (Optimism in the Face of Uncertainty)
- **Định nghĩa:** Đánh giá tích cực các hành động ít được thử để tăng động lực khám phá.
- **Áp dụng:** Cơ chế chung đằng sau Optimistic Initial Values và UCB khiến các hành động ít thử được ưu tiên.



