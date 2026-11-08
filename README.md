# 🍎🍌🍊 Hệ Thống Phân Loại Đa Lớp & Kiểm Định Chất Lượng Hoa Quả (Tươi / Hỏng)
*(Multi-Class Fruit Classification & Freshness Assessment with PyTorch, Transfer Learning & Explainable AI Grad-CAM)*

Dự án Bài tập lớn hoàn chỉnh thực hiện theo quy trình **5 Giai đoạn chuẩn Học thuật & Thực chiến**, tích hợp huấn luyện chịu lỗi trên **Google Colab GPU (Tự động Sao lưu & Auto-Resume Google Drive)** và ứng dụng Web kiểm định thời gian thực trên **VS Code (FastAPI + Grad-CAM XAI)**.

---

## 📂 Cấu Trúc Thư Mục Dự Án (`e:\BTL_PLHoa_qua`)

```text
e:\BTL_PLHoa_qua\
├── .vscode\                      # Cấu hình nhấn F5 chạy Web App ngay trên VS Code
│   ├── launch.json
│   └── tasks.json
├── app\                          # Ứng dụng Web Kiểm định Thực tế (FastAPI + Modern UI)
│   ├── main.py                   # Server Backend & 7 RESTful API Endpoints
│   ├── inference_service.py      # Bộ quản lý 3 mô hình (Thread-safe) & Tạo Grad-CAM
│   ├── fruit_knowledge.json      # CSDL Dinh dưỡng, Bảo quản & Cảnh báo an toàn thực phẩm
│   ├── static\                   # CSS, JS Controller & 6 Ảnh mẫu thử nhanh
│   └── templates\index.html      # Giao diện Dashboard 4 Phân hệ (Tabs)
├── docs\                         # Hồ sơ Thiết kế Chi tiết từng Giai đoạn (Đã kiểm định chéo)
│   ├── Phase1_DeTai_KienTruc.md
│   ├── Phase2_Dataset_EDA_Preprocessing.md
│   ├── Phase3_Colab_Training_Visualization.md
│   └── Phase4_VSCode_Web_Architecture.md
├── models\                       # Nơi chứa 3 file trọng số *_best_model.pth tải về từ Colab
├── notebooks\                    # Notebook chạy 1-Click trên Google Colab GPU
│   └── Colab_Training_Fruit_Classification.ipynb
├── reports\                      # Báo cáo Học thuật 5 Chương & 12 Biểu đồ 300 DPI
│   ├── Bao_Cao_BTL_Hoan_Chinh.md
│   └── figures\                  # Chứa Hình 01 -> Hình 12 (300 DPI)
├── scripts\
│   └── init_local_demo.py        # Khởi tạo ảnh mẫu, hiệu chuẩn trọng số Demo & tạo 12 biểu đồ mẫu
├── src\                          # Thư viện Lõi PyTorch (Dùng chung cho Colab & VS Code)
│   ├── data_pipeline.py          # Làm sạch dữ liệu, chống Data Leakage (dHash) & Xuất EDA
│   ├── models_and_xai.py         # Custom CNN, MobileNetV2, EfficientNet-B0 & Grad-CAM
│   └── trainer_and_eval.py       # Huấn luyện 2 pha có Backup Drive, FSR/FRR & Xuất biểu đồ
└── requirements.txt              # Danh sách thư viện cần cài đặt
```

---

## 🚀 Hướng Dẫn Chạy Nhanh Trên VS Code (Local Web App)

Mở Terminal trong VS Code tại thư mục `e:\BTL_PLHoa_qua` (hoặc nhấn phím **`F5`**):

```powershell
# Bước 1: Cài đặt thư viện
pip install -r requirements.txt

# Bước 2: Khởi tạo 6 ảnh mẫu, hiệu chuẩn trọng số Demo & sinh bộ 12 biểu đồ minh họa
python scripts/init_local_demo.py

# Bước 3: Khởi chạy Web App FastAPI
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

👉 Mở trình duyệt truy cập: **http://127.0.0.1:8000**

---

## ☁️ Hướng Dẫn Huấn Luyện Trên Google Colab GPU (Có Sao Lưu Google Drive)

1. Mở [Google Colab](https://colab.research.google.com/), tải lên file [`notebooks/Colab_Training_Fruit_Classification.ipynb`](file:///e:/BTL_PLHoa_qua/notebooks/Colab_Training_Fruit_Classification.ipynb).
2. Chọn **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**, sau đó bấm **Runtime $\rightarrow$ Run all**.
3. Notebook sẽ tự động:
   - Kết nối Google Drive tại `/content/drive/MyDrive/BTL_PLHoa_qua_Backup/` (chống mất dữ liệu và tự động **Auto-Resume** từ đúng epoch đang dở nếu Colab ngắt kết nối).
   - Tải dataset từ Kaggle qua `kagglehub`, làm sạch kênh `RGBA`, khử trùng lặp `dHash` và chia tập **Stratified Split (70/15/15)**.
   - Huấn luyện cả 3 mô hình (`Custom CNN`, `MobileNetV2`, `EfficientNet-B0`) và xuất trọn bộ **12 biểu đồ 300 DPI Trước & Sau khi Train**.
   - Đóng gói file `BTL_Colab_Results_Bundle.zip` ở Cell cuối cùng.
4. Giải nén `BTL_Colab_Results_Bundle.zip` vào thư mục `e:\BTL_PLHoa_qua` (đưa các file `*_best_model.pth` vào `models/` và 12 ảnh `.png` vào `reports/figures/`), rồi bấm nút **"🔄 Nạp lại Trọng số (.pth)"** trên giao diện Web App!
