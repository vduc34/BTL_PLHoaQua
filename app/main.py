"""
Giai đoạn 4: FastAPI Backend Server cho Hệ thống Phân loại & Đánh giá Chất lượng Hoa quả.
Cung cấp Giao diện Dashboard hiện đại và bộ RESTful API:
  - GET  /                    : Giao diện Web App chính
  - GET  /api/models          : Trạng thái 3 mô hình (`efficientnet_b0`, `mobilenet_v2`, `custom_cnn`)
  - POST /api/reload-models   : Quét lại thư mục `models/` sau khi chép file `.pth` từ Colab về
  - GET  /api/sample-images   : Danh sách ảnh mẫu thử nghiệm nhanh trên giao diện
  - GET  /api/report-figures  : Danh sách 12 biểu đồ báo cáo (300 DPI)
  - POST /api/predict         : Dự đoán 1 ảnh + Top-3 xác suất + Grad-CAM + Tri thức dinh dưỡng/bảo quản
  - POST /api/compare-models  : So sánh trực tiếp cả 3 mô hình trên cùng 1 ảnh đầu vào
  - POST /api/batch-predict   : Kiểm định chất lượng hàng loạt (Batch Inspection) nhiều ảnh cùng lúc
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.inference_service import (
    FruitInferenceRegistry,
    decode_uploaded_image_safe,
)
from src.data_pipeline import load_rgb_image_safe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLES_DIR = APP_DIR / "static" / "samples"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

for folder in (MODELS_DIR, SAMPLES_DIR, FIGURES_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Fruit Quality & Multi-Class AI Inspector",
    description="Hệ thống Phân loại Đa lớp & Đánh giá Chất lượng Hoa quả (Tươi/Hỏng) với Grad-CAM XAI",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
app.mount("/report-figures", StaticFiles(directory=str(FIGURES_DIR)), name="report_figures")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

registry = FruitInferenceRegistry(
    models_dir=MODELS_DIR,
    knowledge_path=APP_DIR / "fruit_knowledge.json",
)


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request) -> HTMLResponse:
    """Render trang Dashboard chính."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"status": registry.get_models_status()},
    )


@app.get("/api/models")
async def get_models_info() -> JSONResponse:
    """Trả về thông tin trạng thái và chỉ số thực nghiệm của 3 mô hình."""
    return JSONResponse(content=registry.get_models_status())


@app.post("/api/reload-models")
async def reload_models_endpoint() -> JSONResponse:
    """Nạp lại các trọng số `.pth` trong thư mục `models/` mà không cần khởi động lại Server."""
    status = registry.reload_all_models()
    return JSONResponse(content={"message": "Đã quét và nạp lại mô hình thành công!", "status": status})


@app.get("/api/sample-images")
async def list_sample_images() -> JSONResponse:
    """Liệt kê các ảnh mẫu có sẵn trong `app/static/samples/` để người dùng click thử ngay."""
    items = []
    for fp in sorted(SAMPLES_DIR.glob("*")):
        if fp.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            items.append(
                {
                    "filename": fp.name,
                    "label_hint": fp.stem,
                    "url": f"/static/samples/{fp.name}",
                }
            )
    return JSONResponse(content={"samples": items})


@app.get("/api/report-figures")
async def list_report_figures() -> JSONResponse:
    """Liệt kê 12 biểu đồ báo cáo (300 DPI) trong `reports/figures/` để xem trực tiếp trên Web."""
    figs = []
    for fp in sorted(FIGURES_DIR.glob("*.png")):
        figs.append(
            {
                "filename": fp.name,
                "title": fp.stem.replace("_", " ").upper(),
                "url": f"/report-figures/{fp.name}",
            }
        )
    return JSONResponse(content={"figures": figs})


@app.post("/api/predict")
async def predict_image_endpoint(
    file: UploadFile = File(default=None),
    sample_filename: str = Form(default=""),
    arch_name: str = Form(default="efficientnet_b0"),
) -> JSONResponse:
    """API nhận diện 1 ảnh (từ Upload/Webcam hoặc từ ảnh mẫu) kèm giải thích Grad-CAM."""
    try:
        if file is not None and file.filename:
            raw_bytes = await file.read()
            pil_img = decode_uploaded_image_safe(raw_bytes)
            source_name = file.filename
        elif sample_filename:
            sample_path = SAMPLES_DIR / Path(sample_filename).name
            if not sample_path.exists():
                raise HTTPException(status_code=404, detail=f"Không tìm thấy ảnh mẫu: {sample_filename}")
            pil_img = load_rgb_image_safe(sample_path)
            source_name = sample_path.name
        else:
            raise HTTPException(status_code=400, detail="Vui lòng tải ảnh lên hoặc chọn một ảnh mẫu.")

        result = registry.predict_single(pil_img=pil_img, arch_name=arch_name)
        result["filename"] = source_name
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {exc}") from exc


@app.post("/api/compare-models")
async def compare_three_models_endpoint(
    file: UploadFile = File(default=None),
    sample_filename: str = Form(default=""),
) -> JSONResponse:
    """API chạy cùng 1 ảnh qua cả 3 mô hình (`efficientnet_b0`, `mobilenet_v2`, `custom_cnn`) để đối chiếu."""
    try:
        if file is not None and file.filename:
            raw_bytes = await file.read()
            pil_img = decode_uploaded_image_safe(raw_bytes)
            source_name = file.filename
        elif sample_filename:
            sample_path = SAMPLES_DIR / Path(sample_filename).name
            if not sample_path.exists():
                raise HTTPException(status_code=404, detail=f"Không tìm thấy ảnh mẫu: {sample_filename}")
            pil_img = load_rgb_image_safe(sample_path)
            source_name = sample_path.name
        else:
            raise HTTPException(status_code=400, detail="Vui lòng tải ảnh lên hoặc chọn một ảnh mẫu.")

        comparison = registry.compare_all_models(pil_img=pil_img)
        comparison["filename"] = source_name
        return JSONResponse(content=comparison)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi so sánh mô hình: {exc}") from exc


@app.post("/api/batch-predict")
async def batch_predict_endpoint(
    files: List[UploadFile] = File(...),
    arch_name: str = Form(default="efficientnet_b0"),
) -> JSONResponse:
    """API kiểm định lô hàng nhiều ảnh cùng lúc (Batch Quality Control) và thống kê tỷ lệ đạt chuẩn."""
    if not files:
        raise HTTPException(status_code=400, detail="Danh sách file trống.")

    batch_results = []
    fresh_count = 0
    rotten_count = 0

    for upload in files[:24]:
        try:
            raw_bytes = await upload.read()
            pil_img = decode_uploaded_image_safe(raw_bytes)
            pred = registry.predict_single(pil_img=pil_img, arch_name=arch_name)
            pred["filename"] = upload.filename or "batch_image.png"
            if pred["status"] == "fresh":
                fresh_count += 1
            else:
                rotten_count += 1
            batch_results.append(pred)
        except Exception as exc:
            batch_results.append({"filename": upload.filename, "error": str(exc)})

    total_valid = fresh_count + rotten_count
    pass_rate = round((fresh_count / total_valid) * 100.0, 2) if total_valid > 0 else 0.0

    return JSONResponse(
        content={
            "arch_name": arch_name,
            "total_processed": total_valid,
            "fresh_count": fresh_count,
            "rotten_count": rotten_count,
            "pass_rate_percent": pass_rate,
            "items": batch_results,
        }
    )
