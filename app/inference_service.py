"""
Giai đoạn 4 (v2.0 - Đã hiệu đính theo Kiểm định Chéo):
Dịch vụ Suy luận Đa Mô hình (Multi-Model Inference Engine) & Giải thích Grad-CAM
trên FastAPI (VS Code). Hỗ trợ nạp tự động các trọng số `*_best_model.pth` từ Google Colab,
phân biệt minh bạch giữa Trọng số Colab thật và Trọng số Demo Local (`is_demo_checkpoint`),
đảm bảo Thread-Safety bằng Lock và xử lý kênh màu RGBA -> RGB chuẩn xác.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.data_pipeline import IMAGENET_MEAN, IMAGENET_STD
from src.models_and_xai import (
    GradCAMExplainer,
    count_parameters,
    create_model,
    load_model_from_checkpoint,
)

DEFAULT_CLASSES = [
    "fresh_apple",
    "fresh_banana",
    "fresh_orange",
    "rotten_apple",
    "rotten_banana",
    "rotten_orange",
]

SUPPORTED_ARCHS = ["efficientnet_b0", "mobilenet_v2", "custom_cnn"]


def pil_to_base64_png(img: Image.Image) -> str:
    """Mã hóa ảnh PIL sang chuỗi Data URL Base64 PNG để hiển thị trực tiếp trên Web UI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def decode_uploaded_image_safe(raw_bytes: bytes) -> Image.Image:
    """
    Giải mã bytes ảnh tải lên từ Web/Webcam và xử lý kênh trong suốt (RGBA / Palette 'P')
    bằng cách ghép lên nền trắng (255, 255, 255) trước khi chuyển về RGB.
    """
    with Image.open(io.BytesIO(raw_bytes)) as img:
        img.load()
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(white_bg, rgba).convert("RGB")
        return img.convert("RGB")


class FruitInferenceRegistry:
    """
    Quản lý vòng đời của cả 3 mô hình (`efficientnet_b0`, `mobilenet_v2`, `custom_cnn`),
    tự động nạp file `{arch}_best_model.pth` trong thư mục `models/` và cung cấp API suy luận Thread-safe.
    """

    def __init__(
        self,
        models_dir: str | Path = "models",
        knowledge_path: str | Path = "app/fruit_knowledge.json",
    ) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()

        self.knowledge_db: Dict[str, Any] = {}
        kp = Path(knowledge_path)
        if kp.exists():
            self.knowledge_db = json.loads(kp.read_text(encoding="utf-8"))

        self.models: Dict[str, torch.nn.Module] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.reload_all_models()

    def reload_all_models(self) -> Dict[str, Any]:
        """Quét thư mục `models/` và nạp tất cả các file `{arch}_best_model.pth`."""
        with self._lock:
            self.models.clear()
            self.metadata.clear()

            for arch in SUPPORTED_ARCHS:
                ckpt_path = self.models_dir / f"{arch}_best_model.pth"
                if ckpt_path.exists():
                    try:
                        model, ckpt = load_model_from_checkpoint(ckpt_path, device=self.device)
                        p_info = count_parameters(model)
                        file_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 2)
                        is_demo = bool(ckpt.get("is_demo_checkpoint", False))
                        self.models[arch] = model
                        self.metadata[arch] = {
                            "arch_name": arch,
                            "loaded_from_checkpoint": not is_demo,
                            "is_demo_checkpoint": is_demo,
                            "checkpoint_file": (
                                f"{ckpt_path.name} (Bản Demo Local)"
                                if is_demo
                                else ckpt_path.name
                            ),
                            "num_classes": ckpt["num_classes"],
                            "class_names": ckpt["class_names"],
                            "idx_to_class": ckpt["idx_to_class"],
                            "img_size": int(ckpt.get("img_size", 224)),
                            "norm_mean": ckpt.get("norm_mean", IMAGENET_MEAN),
                            "norm_std": ckpt.get("norm_std", IMAGENET_STD),
                            "best_epoch": ckpt.get("best_epoch", 0),
                            "best_val_acc": round(float(ckpt.get("best_val_acc", 0.0)), 2),
                            "best_val_f1": round(float(ckpt.get("best_val_f1", 0.0)), 2),
                            "test_metrics": ckpt.get("test_metrics", {}),
                            "total_params": p_info["total_params"],
                            "file_size_mb": file_mb,
                        }
                        continue
                    except Exception as exc:
                        print(f"[WARN] Không thể nạp {ckpt_path}: {exc}")

                model = create_model(arch_name=arch, num_classes=len(DEFAULT_CLASSES), pretrained=False).to(self.device)
                model.eval()
                p_info = count_parameters(model)
                idx_to_class = {i: c for i, c in enumerate(DEFAULT_CLASSES)}
                self.models[arch] = model
                self.metadata[arch] = {
                    "arch_name": arch,
                    "loaded_from_checkpoint": False,
                    "is_demo_checkpoint": True,
                    "checkpoint_file": f"Chưa có {arch}_best_model.pth (Chạy scripts/init_local_demo.py)",
                    "num_classes": len(DEFAULT_CLASSES),
                    "class_names": DEFAULT_CLASSES,
                    "idx_to_class": idx_to_class,
                    "img_size": 224,
                    "norm_mean": IMAGENET_MEAN,
                    "norm_std": IMAGENET_STD,
                    "best_epoch": 0,
                    "best_val_acc": 0.0,
                    "best_val_f1": 0.0,
                    "test_metrics": {},
                    "total_params": p_info["total_params"],
                    "file_size_mb": round((p_info["total_params"] * 4) / (1024 * 1024), 2),
                }

        return self.get_models_status()

    def get_models_status(self) -> Dict[str, Any]:
        """Trả về thông tin trạng thái của cả 3 mô hình cho giao diện Dashboard."""
        return {
            "device": str(self.device),
            "models": self.metadata,
            "default_arch": "efficientnet_b0",
        }

    def _get_domain_info(self, class_label: str) -> Dict[str, Any]:
        """Tra cứu thông tin dinh dưỡng / bảo quản / cảnh báo an toàn từ `fruit_knowledge.json`."""
        if class_label in self.knowledge_db:
            return self.knowledge_db[class_label]

        status = "fresh" if class_label.startswith("fresh_") else "rotten"
        fruit_part = class_label.split("_", 1)[-1].replace("_", " ").title()
        if status == "fresh":
            return {
                "vi_name": f"{fruit_part} Tươi ({class_label})",
                "fruit_vi": f"Quả {fruit_part}",
                "status": "fresh",
                "status_badge": "ĐẠT CHUẨN TƯƠI (SAFE)",
                "calories_per_100g": "45 – 75 kcal",
                "vitamins": "Vitamin C, Vitamin A, Chất xơ & Khoáng chất tự nhiên",
                "storage_temp": "4°C – 8°C",
                "shelf_life": "5 – 14 ngày trong điều kiện mát",
                "quality_assessment": "Bề mặt vỏ nguyên vẹn, màu sắc tươi sáng tự nhiên, không phát hiện dấu hiệu thối nhũn.",
                "recommendation": "Đạt tiêu chuẩn phân phối và tiêu dùng an toàn.",
            }
        return {
            "vi_name": f"{fruit_part} Hỏng ({class_label})",
            "fruit_vi": f"Quả {fruit_part}",
            "status": "rotten",
            "status_badge": "CẢNH BÁO HỎNG (REJECT)",
            "calories_per_100g": "Không sử dụng",
            "vitamins": "Đã biến chất / Nhiễm vi sinh",
            "storage_temp": "Cần cách ly ngay",
            "shelf_life": "0 ngày (Đã hư hỏng)",
            "quality_assessment": "Phát hiện biến đổi sắc tố sẫm màu, mô quả suy thoái hoặc nhiễm nấm mốc.",
            "recommendation": "Cần loại bỏ ngay khỏi lô hàng để đảm bảo an toàn thực phẩm và ngăn lây nhiễm chéo.",
        }

    def predict_single(
        self,
        pil_img: Image.Image,
        arch_name: str = "efficientnet_b0",
        overlay_alpha: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Thực hiện suy luận trên 1 ảnh RGB với mô hình `arch_name`, trích xuất Top-3 xác suất
        và tạo bản đồ nhiệt Grad-CAM (cả ảnh phủ lẫn lớp nhiệt thuần túy).
        """
        arch = arch_name.strip().lower()
        if arch not in self.models:
            arch = "efficientnet_b0"

        meta = self.metadata[arch]
        model = self.models[arch]
        img_size = int(meta["img_size"])
        idx_to_class: Dict[int, str] = meta["idx_to_class"]

        preprocess = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=meta["norm_mean"], std=meta["norm_std"]),
            ]
        )

        resized_rgb = pil_img.convert("RGB").resize((img_size, img_size))
        input_tensor = preprocess(pil_img).unsqueeze(0).to(self.device)

        with self._lock:
            t0 = time.perf_counter()
            explainer = GradCAMExplainer(model=model, arch_name=arch)
            cam_heatmap, pred_idx, probs = explainer.generate_heatmap(input_tensor)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        overlay_pil = GradCAMExplainer.overlay_heatmap_on_image(
            resized_rgb, cam_heatmap, alpha=overlay_alpha
        )
        jet_rgba = matplotlib.colormaps["jet"](cam_heatmap)
        pure_heatmap_pil = Image.fromarray(np.uint8(jet_rgba[:, :, :3] * 255.0))

        sorted_indices = np.argsort(probs)[::-1]
        top3_list: List[Dict[str, Any]] = []
        for rank_i in sorted_indices[:3]:
            lbl = idx_to_class[int(rank_i)]
            info = self._get_domain_info(lbl)
            top3_list.append(
                {
                    "class_idx": int(rank_i),
                    "class_label": lbl,
                    "vi_name": info["vi_name"],
                    "status": info["status"],
                    "confidence_percent": round(float(probs[int(rank_i)]) * 100.0, 2),
                }
            )

        pred_label = idx_to_class[pred_idx]
        domain_info = self._get_domain_info(pred_label)

        fresh_prob_sum = sum(
            float(probs[i]) * 100.0
            for i, name in idx_to_class.items()
            if name.startswith("fresh_")
        )
        rotten_prob_sum = sum(
            float(probs[i]) * 100.0
            for i, name in idx_to_class.items()
            if name.startswith("rotten_")
        )

        return {
            "arch_name": arch,
            "loaded_from_checkpoint": meta["loaded_from_checkpoint"],
            "is_demo_checkpoint": meta["is_demo_checkpoint"],
            "checkpoint_file": meta["checkpoint_file"],
            "predicted_class": pred_label,
            "vi_name": domain_info["vi_name"],
            "fruit_vi": domain_info["fruit_vi"],
            "status": domain_info["status"],
            "status_badge": domain_info["status_badge"],
            "confidence_percent": round(float(probs[pred_idx]) * 100.0, 2),
            "freshness_score_percent": round(fresh_prob_sum, 2),
            "spoilage_risk_percent": round(rotten_prob_sum, 2),
            "latency_ms": latency_ms,
            "top3": top3_list,
            "domain_knowledge": domain_info,
            "images": {
                "original_b64": pil_to_base64_png(resized_rgb),
                "gradcam_overlay_b64": pil_to_base64_png(overlay_pil),
                "gradcam_pure_b64": pil_to_base64_png(pure_heatmap_pil),
            },
        }

    def compare_all_models(self, pil_img: Image.Image) -> Dict[str, Any]:
        """
        Chạy cùng 1 bức ảnh qua cả 3 mô hình (`efficientnet_b0`, `mobilenet_v2`, `custom_cnn`)
        để so sánh trực tiếp Nhãn dự đoán, Độ tự tin %, Bản đồ nhiệt Grad-CAM và Tốc độ (ms).
        """
        comparisons: List[Dict[str, Any]] = []
        for arch in SUPPORTED_ARCHS:
            res = self.predict_single(pil_img, arch_name=arch)
            comparisons.append(res)

        predicted_labels = [c["predicted_class"] for c in comparisons]
        consensus_count = max(predicted_labels.count(lbl) for lbl in set(predicted_labels))

        return {
            "consensus_ratio": f"{consensus_count}/{len(SUPPORTED_ARCHS)}",
            "original_b64": comparisons[0]["images"]["original_b64"],
            "results": comparisons,
        }
