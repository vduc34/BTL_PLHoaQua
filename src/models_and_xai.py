"""
Giai đoạn 3 & 4 (v2.0 - Đã hiệu đính theo Kiểm định Chéo):
Module Kiến trúc 3 Mô hình (Custom CNN, MobileNetV2, EfficientNet-B0),
Quản lý Freeze/Unfreeze 2 Pha (bảo toàn BatchNorm2d), Đóng gói Checkpoint tự thân (.pth)
và Giải thích Mô hình bằng Bản đồ nhiệt Grad-CAM (Tensor-Hook Memory-Safe cho PyTorch 2.x).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ConvBlock(nn.Module):
    """Khối tích chập tiêu chuẩn: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d (inplace=False an toàn cho Grad-CAM)."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Mô hình 1: Custom CNN (Baseline tự thiết kế, huấn luyện từ đầu với Kaiming He Init).
    Cấu trúc: 4 khối ConvBlock (32 -> 64 -> 128 -> 256 kênh) + AdaptiveAvgPool2d + Classifier.
    """

    def __init__(self, num_classes: int, dropout_p: float = 0.4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, pool=True),      # 224 -> 112
            ConvBlock(32, 64, pool=True),     # 112 -> 56
            ConvBlock(64, 128, pool=True),    # 56 -> 28
            ConvBlock(128, 256, pool=False),  # Giữ độ phân giải không gian 28x28 cho Grad-CAM sắc nét
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p, inplace=False),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_p * 0.75, inplace=False),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def create_model(
    arch_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout_p: float = 0.3,
) -> nn.Module:
    """
    Khởi tạo 1 trong 3 kiến trúc thực nghiệm:
      - 'custom_cnn'      : Mạng CNN Baseline tự thiết kế (~1.18M params)
      - 'mobilenet_v2'    : Transfer Learning gọn nhẹ tối ưu tốc độ (~2.24M params)
      - 'efficientnet_b0' : Transfer Learning tích hợp SE-Attention (~4.03M params)
    """
    arch = arch_name.strip().lower()

    if arch == "custom_cnn":
        return CustomCNN(num_classes=num_classes, dropout_p=max(dropout_p, 0.4))

    if arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=False),
            nn.Linear(in_features, num_classes),
        )
        return model

    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=False),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(
        f"Kiến trúc '{arch_name}' không được hỗ trợ. Chọn: 'custom_cnn', 'mobilenet_v2', 'efficientnet_b0'."
    )


def get_target_gradcam_layer(model: nn.Module, arch_name: str) -> nn.Module:
    """
    Trả về tầng tích chập cuối cùng (`target_layer`) để trích xuất bản đồ nhiệt Grad-CAM.
    Cả 3 kiến trúc đều được chuẩn hóa để truy cập qua `model.features[-1]`.
    """
    if hasattr(model, "features"):
        return model.features[-1]
    raise AttributeError(f"Không tìm thấy nhánh 'features' trên mô hình {arch_name}")


def configure_finetuning_stage(
    model: nn.Module,
    arch_name: str,
    stage: int,
    unfreeze_last_blocks: int = 5,
) -> None:
    """
    Thiết lập đóng băng / mở khóa tham số theo Chiến lược Huấn luyện 2 Pha:
      - Stage 1 (Warm-up Head): Đóng băng toàn bộ `model.features`, chỉ mở khóa `model.classifier`.
      - Stage 2 (Fine-tuning): Mở khóa `unfreeze_last_blocks` khối sâu nhất của `model.features`.
      - Với 'custom_cnn': Luôn mở khóa toàn bộ mạng (vì train from scratch).
    """
    arch = arch_name.strip().lower()
    if arch == "custom_cnn":
        for param in model.parameters():
            param.requires_grad = True
        return

    if stage == 1:
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif stage == 2:
        total_blocks = len(model.features)
        cutoff = max(0, total_blocks - unfreeze_last_blocks)
        for idx, block in enumerate(model.features):
            trainable = idx >= cutoff
            for param in block.parameters():
                param.requires_grad = trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError("stage chỉ nhận giá trị 1 (Warm-up) hoặc 2 (Fine-tuning).")


def freeze_bn_stats_if_needed(model: nn.Module) -> None:
    """
    Đặt mọi lớp `BatchNorm2d` có tham số đang bị đóng băng (`requires_grad=False`)
    về chế độ `.eval()` để không làm trôi thống kê `running_mean` và `running_var` của ImageNet.
    Gọi hàm này ngay sau `model.train()` ở mỗi epoch.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if module.weight is not None and not module.weight.requires_grad:
                module.eval()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Đếm tổng số tham số và số tham số đang được huấn luyện."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


# ==============================================================================
# QUẢN LÝ SELF-CONTAINED CHECKPOINT (.PTH) TƯƠNG THÍCH COLAB <-> VS CODE CPU
# ==============================================================================

def save_self_contained_checkpoint(
    save_path: str | Path,
    model: nn.Module,
    arch_name: str,
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    best_epoch: int,
    best_val_acc: float,
    best_val_f1: float,
    img_size: int = 224,
    test_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Đóng gói trọng số mô hình trên CPU kèm đầy đủ từ điển ánh xạ nhãn và thông số chuẩn hóa.
    Đảm bảo khi tải về VS Code (Windows CPU) không bao giờ gặp lỗi thiếu CUDA hay lệch thứ tự lớp.
    """
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    payload = {
        "arch_name": arch_name.strip().lower(),
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "img_size": img_size,
        "norm_mean": IMAGENET_MEAN,
        "norm_std": IMAGENET_STD,
        "model_state_dict": cpu_state_dict,
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_f1": float(best_val_f1),
        "test_metrics": test_metrics or {},
    }
    torch.save(payload, out_path)
    return out_path


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Tải mô hình an toàn trên cả CPU lẫn GPU từ file `.pth` chuẩn Self-Contained.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch_name = ckpt["arch_name"]
    num_classes = ckpt["num_classes"]

    model = create_model(arch_name=arch_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    raw_idx_to_class = ckpt.get("idx_to_class", {})
    ckpt["idx_to_class"] = {int(k): v for k, v in raw_idx_to_class.items()}
    return model, ckpt


# ==============================================================================
# GIẢI THÍCH MÔ HÌNH BẰNG GRAD-CAM (TENSOR-HOOK AN TOÀN TRÊN PYTORCH 2.X)
# ==============================================================================

class GradCAMExplainer:
    """
    Trích xuất bản đồ nhiệt Grad-CAM trên tầng tích chập cuối cùng (`target_layer`).
    Sử dụng `output.register_hook` (Tensor-level hook) ngay trong `forward_hook`
    để tránh hoàn toàn lỗi `BackwardHookFunctionBackward inplace view` của PyTorch 2.x,
    đồng thời gỡ bỏ hook trong khối `try...finally` chống rò rỉ RAM trên FastAPI.
    """

    def __init__(self, model: nn.Module, arch_name: str = "model") -> None:
        self.model = model
        self.arch_name = arch_name
        self.target_layer = get_target_gradcam_layer(model, arch_name)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Tính toán bản đồ nhiệt Grad-CAM cho 1 ảnh đầu vào `input_tensor` (1, 3, H, W).
        Trả về:
          - cam_heatmap : Mảng 2D float32 kích thước (H, W) chuẩn hóa trong đoạn [0, 1]
          - pred_idx    : Chỉ số lớp dự đoán cao nhất
          - probs       : Vector xác suất Softmax của tất cả các lớp
        """
        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def forward_hook(_module: nn.Module, _inp: Any, output: torch.Tensor) -> None:
            activations.append(output.detach())
            if output.requires_grad:
                output.register_hook(lambda grad: gradients.append(grad.detach()))

        f_handle = self.target_layer.register_forward_hook(forward_hook)

        prev_grad_states = {p: p.requires_grad for p in self.target_layer.parameters()}
        for p in self.target_layer.parameters():
            p.requires_grad_(True)

        try:
            self.model.eval()
            self.model.zero_grad(set_to_none=True)

            with torch.enable_grad():
                x = input_tensor.clone().detach().requires_grad_(True)
                logits = self.model(x)
                probs_tensor = F.softmax(logits, dim=1)
                probs = probs_tensor[0].detach().cpu().numpy()
                pred_idx = int(np.argmax(probs))

                cls_idx = pred_idx if target_class_idx is None else int(target_class_idx)
                score = logits[0, cls_idx]
                score.backward()

            act = activations[0][0]   # (C, h, w)
            grad = gradients[0][0]    # (C, h, w)

            # Trọng số alpha_k = Global Average Pooling của đạo hàm theo không gian (h, w)
            weights = torch.mean(grad, dim=(1, 2), keepdim=True)  # (C, 1, 1)
            cam = torch.sum(weights * act, dim=0)                 # (h, w)
            cam = F.relu(cam)

            cam_min, cam_max = cam.min(), cam.max()
            if (cam_max - cam_min) > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            h_img, w_img = input_tensor.shape[2], input_tensor.shape[3]
            cam_resized = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(h_img, w_img),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            return cam_resized.cpu().numpy(), pred_idx, probs
        finally:
            f_handle.remove()
            for p, old_state in prev_grad_states.items():
                p.requires_grad_(old_state)
            self.model.zero_grad(set_to_none=True)

    @staticmethod
    def overlay_heatmap_on_image(
        rgb_image: Image.Image | np.ndarray,
        cam_heatmap: np.ndarray,
        alpha: float = 0.45,
    ) -> Image.Image:
        """
        Phủ bản đồ nhiệt Grad-CAM (bảng màu Jet chuẩn Matplotlib 3.8+) lên trên ảnh RGB gốc.
        """
        if isinstance(rgb_image, Image.Image):
            img_np = np.asarray(rgb_image.convert("RGB"), dtype=np.float32) / 255.0
        else:
            img_np = np.clip(rgb_image.astype(np.float32), 0.0, 1.0)

        h, w = img_np.shape[:2]
        if cam_heatmap.shape[:2] != (h, w):
            cam_pil = Image.fromarray(np.uint8(cam_heatmap * 255)).resize((w, h), Image.Resampling.BILINEAR)
            cam_heatmap = np.asarray(cam_pil, dtype=np.float32) / 255.0

        jet_colormap = matplotlib.colormaps["jet"]
        heatmap_rgb = jet_colormap(cam_heatmap)[:, :, :3]

        blended = (1.0 - alpha) * img_np + alpha * heatmap_rgb
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)
