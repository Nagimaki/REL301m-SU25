# Bài học 1: Vấn đề K-Armed Bandit

## 1. Phần thưởng (Reward)
- **Định nghĩa:** Giá trị nhận được ngay sau khi thực hiện một hành động, phản ánh mức độ “tốt” hay “hiệu quả” của hành động đó.
- **Ký hiệu:**  
  ![](https://latex.codecogs.com/png.latex?R_t)  
  là phần thưởng tại thời điểm ![](https://latex.codecogs.com/png.latex?t).
- **Ví dụ bác sĩ:**
  - Bác sĩ có 3 loại thuốc (A, B, C) điều trị một bệnh.
  - Khi kê thuốc A, bệnh nhân hồi phục nhanh →  
    ![](https://latex.codecogs.com/png.latex?R_t%20%3D%2B10).
  - Nếu bệnh nhân gặp tác dụng phụ nhẹ →  
    ![](https://latex.codecogs.com/png.latex?R_t%20%3D%20-2).

## 2. Tính chất thời gian của vấn đề Bandit (Temporal Property)
- **Định nghĩa:** Chuỗi hành động và phần thưởng xảy ra tuần tự theo thời gian; lựa chọn hiện tại ảnh hưởng đến kết quả và chiến lược tương lai.
- **Biểu diễn:**  
  ![](https://latex.codecogs.com/png.latex?A_1%2C%20R_1%2C%20A_2%2C%20R_2%2C%20%5Cdots%2C%20A_t%2C%20R_t)
- **Ví dụ bác sĩ:**
  - Ngày 1 bác sĩ kê thuốc A → bệnh nhân cải thiện nhẹ (  
    ![](https://latex.codecogs.com/png.latex?R_1%20%3D%20%2B5) ).
  - Ngày 2, dựa vào kết quả hôm trước, bác sĩ đổi sang thuốc B → bệnh nhân hồi phục tốt hơn (  
    ![](https://latex.codecogs.com/png.latex?R_2%20%3D%20%2B8) ).

## 3. K-Armed Bandit
- **Định nghĩa:** Mô hình với ![](https://latex.codecogs.com/png.latex?K) hành động (arms), mỗi hành động cho phần thưởng ngẫu nhiên theo phân phối khác nhau; mục tiêu tối đa hóa tổng phần thưởng.
- **Mục tiêu:**  
  ![](https://latex.codecogs.com/png.latex?%5Cmax%20%5Csum_%7Bt%3D1%7D%5ET%20R_t)
- **Ví dụ bác sĩ:**
  - Bác sĩ có 4 phương pháp điều trị: thuốc A, thuốc B, phẫu thuật, vật lý trị liệu.
  - Mỗi phương pháp cho phần thưởng khác nhau tùy bệnh nhân (ví dụ: phẫu thuật hồi phục nhanh nhưng rủi ro cao).
  - Bác sĩ phải thử nghiệm và chọn phương pháp tốt nhất cho nhóm bệnh nhân.

## 4. Giá trị hành động (Action Value)
- **Định nghĩa:** Kỳ vọng phần thưởng nếu luôn chọn hành động đó.
- **Giá trị thực:**  
  ![](https://latex.codecogs.com/png.latex?q_%2A(a)%20%3D%20%5Cmathbb%7BE%7D%5BR_t%20%7C%20A_t%20%3D%20a%5D)
- **Ước lượng (Sample Average):**  
  ![](https://latex.codecogs.com/png.latex?Q_n(a)%20%3D%20%5Cfrac%7Br_1%20%2B%20r_2%20%2B%20%5Cdots%20%2B%20r_n%7D%7BN(a)%7D)  
  Trong đó ![](https://latex.codecogs.com/png.latex?N(a)) là số lần đã chọn hành động a.
- **Ví dụ bác sĩ:**
  - Bác sĩ đã kê thuốc A cho 5 bệnh nhân, nhận phần thưởng +6, +7, +5, +8, +6.  
    ![](https://latex.codecogs.com/png.latex?Q(A)%20%3D%20%5Cfrac%7B6%2B7%2B5%2B8%2B6%7D%7B5%7D%20%3D%206.4)

---

# Bài học 2: Học gì? Ước lượng giá trị hành động

## 1. Phương pháp ước lượng giá trị hành động
- **Định nghĩa:** Cách tính toán giá trị kỳ vọng của từng hành động dựa trên phần thưởng thu thập từ kinh nghiệm.
- **Công thức Sample Average:**  
  ![](https://latex.codecogs.com/png.latex?Q_n(a)%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5En%20r_i%7D%7BN(a)%7D)
- **Ví dụ bác sĩ:**
  - Kê thuốc B cho 3 bệnh nhân, phần thưởng +7, +6, +8.  
    ![](https://latex.codecogs.com/png.latex?Q(B)%20%3D%20%5Cfrac%7B7%2B6%2B8%7D%7B3%7D%20%3D%207)

## 2. Khám phá và khai thác (Exploration vs. Exploitation)
- **Định nghĩa:**
  - **Khai thác:** Chọn hành động có giá trị cao nhất hiện tại để tối đa hóa phần thưởng ngắn hạn.
  - **Khám phá:** Thử hành động mới để thu thập thông tin, tối ưu hóa lâu dài.
- **Ví dụ bác sĩ:** Thường kê thuốc B (khai thác) nhưng thỉnh thoảng thử thuốc C (khám phá) để phát hiện hiệu quả tiềm năng.

## 3. Chọn hành động tham lam theo giá trị hành động (Greedy)
- **Định nghĩa:** Luôn chọn hành động có ước lượng giá trị ![](https://latex.codecogs.com/png.latex?Q(a)) cao nhất.
- **Công thức:**  
  ![](https://latex.codecogs.com/png.latex?A_t%20%3D%20%5Carg%5Cmax_a%20Q(a))
- **Ví dụ bác sĩ:** Giữa thuốc A (6.4), B (7), C (5) → luôn chọn thuốc B.

## 4. Học trực tuyến (Online Learning)
- **Định nghĩa:** Cập nhật giá trị hành động ngay khi nhận được phần thưởng mới, không phải chờ hết tập dữ liệu.

## 5. Phương pháp trung bình mẫu đơn giản (Sample Average Method)
- **Cập nhật đệ quy:**  
  ![](https://latex.codecogs.com/png.latex?Q_%7Bn%2B1%7D%20%3D%20Q_n%20%2B%20%5Cfrac%7B1%7D%7Bn%7D%28R_n%20-%20Q_n%29)
- **Ví dụ bác sĩ:**  
  - ![](https://latex.codecogs.com/png.latex?Q_3%20%3D%206.4), ![](https://latex.codecogs.com/png.latex?R_4%20%3D%209)  
  - ![](https://latex.codecogs.com/png.latex?Q_4%20%3D%206.4%20%2B%20%5Cfrac%7B1%7D%7B4%7D(9-6.4)%20%3D%207.05)

## 6. Phương trình cập nhật trực tuyến tổng quát (General Update)
- **Công thức với learning rate ![](https://latex.codecogs.com/png.latex?%5Calpha):**  
  ![](https://latex.codecogs.com/png.latex?Q_%7Bn%2B1%7D%20%3D%20Q_n%20%2B%20%5Calpha%20(R_n%20-%20Q_n)%2C%200%20%3C%5Calpha%20%5Cleq%201)
- **Ví dụ bác sĩ:**  
  - ![](https://latex.codecogs.com/png.latex?Q_n%20%3D%206.4), ![](https://latex.codecogs.com/png.latex?R%20%3D%209), ![](https://latex.codecogs.com/png.latex?%5Calpha%20%3D%200.1)  
  - ![](https://latex.codecogs.com/png.latex?Q'%20%3D%206.4%20%2B%200.1(9-6.4)%20%3D%206.66)

## 7. Bước cập nhật không đổi cho tình trạng không ổn định
- **Định nghĩa:** Dùng ![](https://latex.codecogs.com/png.latex?%5Calpha) cố định để nhanh chóng thích nghi với môi trường thay đổi (non-stationary).
- **Công thức:**  
  ![](https://latex.codecogs.com/png.latex?Q_%7Bn%2B1%7D%20%3D%20Q_n%20%2B%20%5Calpha(R_n%20-%20Q_n))
- **Ví dụ bác sĩ:** Virus đột biến, hiệu quả thuốc thay đổi → chọn ![](https://latex.codecogs.com/png.latex?%5Calpha%20%3D%200.1) cố định để cập nhật nhanh.

---

# Bài học 3: Sự đánh đổi giữa khám phá và khai thác

## 1. Epsilon-Greedy (ε-Greedy)
- **Định nghĩa:** Với xác suất ![](https://latex.codecogs.com/png.latex?1-%5Cepsilon) chọn hành động tốt nhất, với xác suất ![](https://latex.codecogs.com/png.latex?%5Cepsilon) chọn ngẫu nhiên.
- **Công thức:**  
  ![](https://latex.codecogs.com/png.latex?A_t%20%3D%20%5Cbegin%7Bcases%7D%20%5Carg%5Cmax_a%20Q(a)%2C%20%26%20%5Ctext%7Bv%E1%BB%9Bi%20x%C3%A1c%20su%E1%BA%A5t%7D%201-%5Cepsilon%5C%5C%20%5Ctext%7Brandom%20action%7D%2C%20%26%20%5Ctext%7Bv%E1%BB%9Bi%20x%C3%A1c%20su%E1%BA%A5t%7D%5Cepsilon%20%5Cend%7Bcases%7D)
- **Ví dụ bác sĩ:** ![](https://latex.codecogs.com/png.latex?%5Cepsilon%20%3D%200.1) → 90% chọn thuốc tốt nhất, 10% thử thuốc khác.

## 2. Lợi ích ngắn hạn của khai thác và dài hạn của khám phá
- **Khai thác:** Tối đa phần thưởng ngay lập tức.
- **Khám phá:** Phát hiện hành động tốt hơn cho tương lai.
- **Ví dụ bác sĩ:** Khai thác thuốc B (7) ngắn hạn, khám phá thuốc C có thể là 9 dài hạn.

## 3. Giá trị khởi tạo lạc quan (Optimistic Initial Values)
- **Định nghĩa:** Gán giá trị ước lượng cao khởi đầu cho mọi hành động để khuyến khích khám phá.
- **Ví dụ bác sĩ:** Ban đầu đặt ![](https://latex.codecogs.com/png.latex?Q(A)%20%3D%20Q(B)%20%3D%20Q(C)%20%3D%2010) để đảm bảo thử hết cả ba thuốc.

## 4. Lợi ích của giá trị khởi tạo lạc quan
- Đảm bảo mọi hành động được thử ít nhất một lần, tránh bỏ sót lựa chọn tốt.

## 5. Phê bình giá trị khởi tạo lạc quan
- Sau khi đủ khám phá, các giá trị hội tụ, giá trị khởi tạo lạc quan không còn tác động.

## 6. Phương pháp UCB (Upper Confidence Bound)
- **Định nghĩa:** Chọn ![](https://latex.codecogs.com/png.latex?a) tối đa hóa tổng giá trị ước lượng và mức độ không chắc chắn.
- **Công thức:**  
  ![](https://latex.codecogs.com/png.latex?A_t%20%3D%20%5Carg%5Cmax_a%20%5Cbigl(Q(a)%20%2B%20c%20%5Csqrt%7B%5Cfrac%7B%5Cln%20t%7D%7BN(a)%7D%7D%5Cbigr))
- **Ví dụ bác sĩ:** Thuốc mới thử ít lần có phần bù lớn → được thử thêm.

## 7. Lạc quan trong sự không chắc chắn (Optimism in the Face of Uncertainty)
- **Định nghĩa:** Đánh giá tích cực các hành động ít được thử để tăng động lực khám phá.
- **Áp dụng:** Cơ chế chung đằng sau Optimistic Initial Values và UCB khiến các hành động ít thử được ưu tiên.

