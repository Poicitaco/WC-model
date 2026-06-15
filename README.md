---
title: Dự Đoán World Cup 2026 bằng Trí Tuệ Nhân Tạo (AI)
emoji: ⚽
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🏆 Hệ Thống Phân Tích & Dự Đoán World Cup 2026 bằng Machine Learning

Chào mừng đến với dự án **Dự Đoán World Cup 2026**. Đây là một ứng dụng web kết hợp khoa học dữ liệu (Data Science) và trí tuệ nhân tạo (Machine Learning) để mô phỏng và đưa ra dự báo kết quả của giải đấu bóng đá hấp dẫn nhất hành tinh.

![World Cup Banner](https://upload.wikimedia.org/wikipedia/en/thumb/f/fa/2026_FIFA_World_Cup.svg/1200px-2026_FIFA_World_Cup.svg.png)

---

## ✨ Tính Năng Chính

Dự án cung cấp một bộ công cụ phân tích bóng đá toàn diện với các tính năng:
- **📊 Mô Phỏng & Phân Tích Kỹ Thuật:** Hiển thị 21+ biểu đồ khoa học dữ liệu chuyên sâu (Ma trận nhầm lẫn, Feature Importance, Hat Graph, Radar Chart, v.v.) kèm theo phân tích chi tiết cho từng loại thuật toán (XGBoost, Random Forest).
- **📅 Lịch Thi Đấu Tích Hợp:** Lấy dữ liệu lịch thi đấu thực tế tự động từ API (football-data.org), tự động cập nhật khi có biến động.
- **🔮 Lịch Dự Đoán AI:** Tự động áp dụng mô hình Machine Learning (đã được fine-tune) vào lịch thi đấu để đưa ra dự báo trực tiếp về Tỷ lệ Thắng / Hòa / Thua.
- **🏆 Sơ Đồ Knockout (Vòng Loại Trực Tiếp):** Vẽ sơ đồ luồng giải đấu từ Vòng bảng đến Vòng 32 đội, 16 đội, Tứ kết, Bán kết và Chung kết, tìm ra nhà Vô địch có xác suất cao nhất.
- **🎤 Chế Độ Thuyết Trình (Presentation):** Tích hợp sẵn HTML Presentation Slideshow chuyên nghiệp, biến toàn bộ nghiên cứu thành tài liệu thuyết trình mượt mà (hỗ trợ chuyển slide tự động).
- **📸 Photobooth:** Trải nghiệm tương tác với hệ thống chụp hình Photobooth chủ đề World Cup 2026 có tính năng quản lý khung hình.

---

## 🛠️ Công Nghệ Sử Dụng

- **Backend:** Python, Flask, Pandas, Scikit-Learn, XGBoost
- **Frontend:** HTML5, Tailwind CSS, JavaScript thuần (Vanilla JS), FontAwesome, Phosphor Icons
- **Deployment:** 
  - Tích hợp CI/CD tự động lên nền tảng **Vercel** (`vercel.json`).
  - Hỗ trợ triển khai thông qua **Docker** (có sẵn `Dockerfile`) dành cho Hugging Face Spaces hoặc AWS/GCP.

---

## 📂 Cấu Trúc Dự Án

```bash
WC2026_FINAL/
├── app.py                      # Flask Application Backend (API, Routing, Data Processing)
├── data_processed/             # Dữ liệu CSV đã qua xử lý (Team data, Elo, Lịch thi đấu)
├── templates/                  # Frontend HTML Views
│   ├── index.html              # Giao diện chính của hệ thống dự đoán
│   ├── photobooth.html         # Chức năng chụp ảnh Photobooth
│   └── admin_photobooth.html   # Quản lý khung ảnh
├── static/                     # Tài nguyên tĩnh
│   ├── anh/                    # Chứa 21+ ảnh biểu đồ Data Science (không dùng LFS)
│   ├── images/                 # Hình ảnh cờ quốc gia, logo giải đấu
│   └── presentation/           # Source code cho chế độ Slide Thuyết trình
├── outputs/                    # Log dự đoán và các bảng kết quả xuất ra dạng file
├── models/                     # Các file trọng số mô hình đã huấn luyện (.pkl)
├── requirements.txt            # Danh sách thư viện Python phụ thuộc
└── vercel.json                 # Cấu hình triển khai lên nền tảng Vercel
```

---

## 🚀 Hướng Dẫn Cài Đặt (Chạy Local)

Nếu bạn muốn chạy thử nghiệm dự án này ngay trên máy tính cá nhân của mình:

### 1. Chuẩn bị môi trường
Yêu cầu hệ thống phải cài đặt sẵn **Python 3.8+** và công cụ quản lý package `pip`. Khuyên dùng môi trường ảo (Virtual Environment) để cài đặt thư viện.

### 2. Cài đặt các thư viện phụ thuộc
Di chuyển vào thư mục gốc của dự án và chạy lệnh sau:
```bash
pip install -r requirements.txt
```

### 3. Khởi chạy Server
Khởi động Flask Server bằng lệnh:
```bash
python app.py
```
Sau đó, hãy mở trình duyệt web và truy cập địa chỉ: [http://127.0.0.1:5000](http://127.0.0.1:5000) để trải nghiệm.

---

## ☁️ Hướng Dẫn Triển Khai (Deployment)

### Lựa chọn 1: Triển khai lên Vercel (Khuyên dùng)
Dự án đã được cấu hình sẵn để chạy cực nhẹ trên Vercel qua Serverless Functions.
1. Khởi tạo tài khoản Vercel và kết nối với Github Repository của bạn.
2. Tại Vercel Dashboard, thêm dự án mới. Vercel sẽ tự động đọc file `vercel.json` và cấu hình Python Runtime.
3. Nhấn **Deploy**. *Lưu ý: Tất cả hình ảnh trong thư mục `static/anh/` đã được gỡ LFS nên Vercel sẽ render bình thường không bị lỗi.*

### Lựa chọn 2: Triển khai qua Docker
Nếu muốn host trên server riêng (VPS) hoặc Hugging Face Spaces:
```bash
docker build -t wc2026-simulator .
docker run -p 7860:7860 wc2026-simulator
```
Truy cập qua cổng đã map (mặc định là cổng `7860`).

---

## 👨‍💻 Đóng Góp
Mọi ý kiến đóng góp nhằm tối ưu thuật toán hoặc cải thiện giao diện đều được hoan nghênh. Vui lòng tạo *Pull Request* hoặc mở *Issues* trên Github.

---
*Dự án phục vụ mục đích nghiên cứu học thuật và học tập. Xác suất dự đoán được tính toán từ thuật toán máy học, không mang tính chất cổ xúy cá cược.*
