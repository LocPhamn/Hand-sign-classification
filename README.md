# Hand Sign Classification

Hand Sign Classification là một dự án sử dụng học máy (machine learning) để nhận diện cử chỉ tay từ hình ảnh, với 6 nhãn: `Scale`, `Point`, `Other`, `None`, `Loupe`, và `Drag`. Dự án này có thể được áp dụng trong các ứng dụng điều khiển không chạm, nhận diện ngôn ngữ ký hiệu, hoặc tương tác người-máy.

---

## 1. Dataset
Dữ liệu được chuẩn bị bao gồm các hình ảnh chứa bàn tay với các tư thế tương ứng. Bộ dữ liệu được phân chia thành các nhãn như sau:
- **Scale**: Tay thể hiện tư thế giãn hoặc co (như phóng to/thu nhỏ).
- **Point**: Tay chỉ vào một hướng hoặc đối tượng.
- **Other**: Các tư thế khác không thuộc các nhóm chính.
- **None**: Không có bàn tay trong hình.
- **Loupe**: Tay tạo hình giống như kính lúp (ngón cái và ngón trỏ tạo vòng tròn).
- **Drag**: Tay thể hiện hành động kéo hoặc di chuyển.

---

## 2. Mục tiêu
Xây dựng một mô hình phân loại hình ảnh chính xác để:
- Phân biệt 6 nhãn trên từ ảnh đầu vào.
- Hỗ trợ các hệ thống tương tác dựa trên cử chỉ tay.

---

## 3. Công nghệ
Dự án sử dụng các công cụ và thư viện chính sau:
- **Python**: Ngôn ngữ lập trình chính.
- **TensorFlow / PyTorch**: Xây dựng và huấn luyện mô hình.
- **OpenCV**: Xử lý hình ảnh và phát hiện bàn tay.
- **Matplotlib / Seaborn**: Trực quan hóa dữ liệu và kết quả.
- **Scikit-learn**: Đánh giá hiệu suất mô hình.

---

## 4. Cách sử dụng

### 4.1 Cài đặt
Clone repository và cài đặt các phụ thuộc:

```bash
git clone https://github.com/LocPhamn/Hand-sign-classification.git
cd Hand-sign-classification
pip install -r requirements.txt
