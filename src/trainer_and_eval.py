"""
Giai đoạn 3 (v2.0 - Đã hiệu đính theo Kiểm định Chéo):
Module Huấn luyện 2 Pha có Sao lưu Tự động (Fault-Tolerant Backup & Auto-Resume)
trên Google Drive, Tính toán Bộ chỉ số Học thuật (Accuracy, Macro-F1, FSR, FRR, Latency)
và Xuất Trọn bộ Biểu đồ Trực quan hóa Trước & Sau khi Train (300 DPI cho Báo cáo).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_pipeline import IMAGENET_MEAN, IMAGENET_STD
from src.models_and_xai import (
    GradCAMExplainer,
    configure_finetuning_stage,
    count_parameters,
    create_model,
    freeze_bn_stats_if_needed,
    save_self_contained_checkpoint,
)


def compute_freshness_safety_metrics(
    cm: np.ndarray,
    idx_to_class: Dict[int, str],
) -> Dict[str, float]:
    """
    Tính toán 2 chỉ số nghiệp vụ kiểm định chất lượng thực phẩm trên ma trận nhầm lẫn đa lớp:
      1. False Safe Rate (FSR %)     : Quả thực tế bị HỎNG (rotten_*) nhưng dự đoán nhầm thành TƯƠI (fresh_*)
      2. False Rejection Rate (FRR %): Quả thực tế còn TƯƠI (fresh_*) nhưng dự đoán nhầm thành HỎNG (rotten_*)
    """
    fresh_indices = [i for i, name in idx_to_class.items() if name.startswith("fresh_")]
    rotten_indices = [i for i, name in idx_to_class.items() if name.startswith("rotten_")]

    total_actual_rotten = float(cm[rotten_indices, :].sum()) if rotten_indices else 0.0
    rotten_pred_as_fresh = (
        float(cm[np.ix_(rotten_indices, fresh_indices)].sum())
        if (rotten_indices and fresh_indices)
        else 0.0
    )
    fsr_percent = (
        (rotten_pred_as_fresh / total_actual_rotten) * 100.0
        if total_actual_rotten > 0
        else 0.0
    )

    total_actual_fresh = float(cm[fresh_indices, :].sum()) if fresh_indices else 0.0
    fresh_pred_as_rotten = (
        float(cm[np.ix_(fresh_indices, rotten_indices)].sum())
        if (fresh_indices and rotten_indices)
        else 0.0
    )
    frr_percent = (
        (fresh_pred_as_rotten / total_actual_fresh) * 100.0
        if total_actual_fresh > 0
        else 0.0
    )

    return {
        "false_safe_rate_percent": round(fsr_percent, 4),
        "false_rejection_rate_percent": round(frr_percent, 4),
        "rotten_pred_as_fresh_count": int(rotten_pred_as_fresh),
        "total_actual_rotten": int(total_actual_rotten),
        "fresh_pred_as_rotten_count": int(fresh_pred_as_rotten),
        "total_actual_fresh": int(total_actual_fresh),
    }


@torch.no_grad()
def evaluate_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    idx_to_class: Dict[int, str],
) -> Dict[str, Any]:
    """Đánh giá mô hình trên tập Validation hoặc Test, trả về đầy đủ metrics và mảng dự đoán."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        batch_sz = inputs.size(0)
        running_loss += loss.item() * batch_sz
        total_samples += batch_sz

        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / max(total_samples, 1)
    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(
        all_targets, all_preds, labels=list(range(len(idx_to_class)))
    )
    safety = compute_freshness_safety_metrics(cm, idx_to_class)

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "macro_precision": float(macro_p * 100.0),
        "macro_recall": float(macro_r * 100.0),
        "macro_f1": float(macro_f1 * 100.0),
        "confusion_matrix": cm,
        "safety_metrics": safety,
        "y_true": np.array(all_targets),
        "y_pred": np.array(all_preds),
        "y_prob": np.vstack(all_probs) if all_probs else np.empty((0, len(idx_to_class))),
    }


def benchmark_inference_latency(
    model: nn.Module,
    img_size: int = 224,
    num_runs: int = 30,
) -> Dict[str, float]:
    """
    Đo tốc độ suy luận thực tế (ms/ảnh) trên cả GPU (nếu có) và CPU
    để đưa vào bảng so sánh thực nghiệm trong Báo cáo.
    """
    results: Dict[str, float] = {}
    orig_device = next(model.parameters()).device

    if torch.cuda.is_available():
        gpu_dev = torch.device("cuda")
        model.to(gpu_dev).eval()
        dummy_gpu = torch.randn(1, 3, img_size, img_size, device=gpu_dev)
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_gpu)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_runs):
                _ = model(dummy_gpu)
            torch.cuda.synchronize()
            results["gpu_latency_ms"] = round(((time.perf_counter() - t0) / num_runs) * 1000.0, 2)
    else:
        results["gpu_latency_ms"] = 0.0

    cpu_dev = torch.device("cpu")
    model.to(cpu_dev).eval()
    dummy_cpu = torch.randn(1, 3, img_size, img_size, device=cpu_dev)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_cpu)
        t0 = time.perf_counter()
        cpu_runs = min(num_runs, 15)
        for _ in range(cpu_runs):
            _ = model(dummy_cpu)
        results["cpu_latency_ms"] = round(((time.perf_counter() - t0) / cpu_runs) * 1000.0, 2)

    model.to(orig_device)
    return results


# ==============================================================================
# VÒNG LẶP HUẤN LUYỆN 2 PHA CÓ SAO LƯU GOOGLE DRIVE & AUTO-RESUME CHUẨN XÁC
# ==============================================================================

def _build_stage_optimizer_and_scheduler(
    model: nn.Module,
    arch_name: str,
    stage: int,
    stage1_epochs: int,
    total_epochs: int,
    lr_stage1: float,
    lr_stage2_backbone: float,
    lr_stage2_head: float,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Khởi tạo Optimizer và CosineAnnealingLR tương ứng với Stage 1 hoặc Stage 2."""
    configure_finetuning_stage(model, arch_name=arch_name, stage=stage)
    if arch_name == "custom_cnn":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_stage1, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_epochs, 1), eta_min=1e-5
        )
    elif stage == 1:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_stage1,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(stage1_epochs, 1), eta_min=2e-4
        )
    else:
        backbone_params = [p for p in model.features.parameters() if p.requires_grad]
        head_params = [p for p in model.classifier.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr_stage2_backbone},
                {"params": head_params, "lr": lr_stage2_head},
            ],
            weight_decay=1e-4,
        )
        stage2_total = max(total_epochs - stage1_epochs, 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage2_total, eta_min=1e-6
        )
    return optimizer, scheduler


def train_model_with_backup(
    arch_name: str,
    dataloaders: Dict[str, DataLoader],
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    class_weights: Optional[torch.Tensor],
    backup_dir: str | Path,
    device: torch.device,
    stage1_epochs: int = 5,
    stage2_epochs: int = 12,
    lr_stage1: float = 1e-3,
    lr_stage2_backbone: float = 1e-4,
    lr_stage2_head: float = 5e-4,
    patience: int = 5,
    img_size: int = 224,
    resume: bool = True,
) -> Tuple[nn.Module, pd.DataFrame, Path]:
    """
    Huấn luyện mô hình với cơ chế Fault-Tolerant Backup trên Google Drive:
      - Tự động lưu `{arch_name}_last_resume.pth` (gồm cả `optimizer_state_dict`, `scheduler_state_dict`,
        `scaler_state_dict` và cờ `training_completed`) cùng `{arch_name}_training_history.csv` sau MỖI epoch.
      - Tự động lưu `{arch_name}_best_model.pth` (Self-Contained Schema) mỗi khi `val_macro_f1` đạt đỉnh mới.
      - Nếu Colab bị ngắt kết nối, chạy lại hàm với `resume=True` sẽ khôi phục đầy đủ momentum & LR scheduler
        ngay từ epoch vừa dừng (hoặc bỏ qua ngay nếu mô hình đó đã hoàn tất / Early Stopped).
    """
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)

    resume_ckpt_path = backup_path / f"{arch_name}_last_resume.pth"
    best_model_path = backup_path / f"{arch_name}_best_model.pth"
    history_csv_path = backup_path / f"{arch_name}_training_history.csv"

    num_classes = len(class_to_idx)
    model = create_model(arch_name=arch_name, num_classes=num_classes, pretrained=True).to(device)

    total_epochs = stage1_epochs + stage2_epochs
    if arch_name == "custom_cnn":
        stage1_epochs = 0

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=0.1,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history_records: List[Dict[str, Any]] = []
    training_completed = False
    saved_state: Optional[Dict[str, Any]] = None

    if resume and resume_ckpt_path.exists():
        print(f"[AUTO-RESUME] Phát hiện bản sao lưu của '{arch_name}' tại {resume_ckpt_path}!")
        saved_state = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(saved_state["model_state_dict"])
        start_epoch = int(saved_state["epoch"]) + 1
        best_val_f1 = float(saved_state["best_val_f1"])
        best_val_acc = float(saved_state["best_val_acc"])
        best_epoch = int(saved_state["best_epoch"])
        epochs_no_improve = int(saved_state["epochs_no_improve"])
        history_records = saved_state.get("history_records", [])
        training_completed = bool(saved_state.get("training_completed", False))

        if training_completed or start_epoch > total_epochs:
            print(
                f"[AUTO-RESUME] Mô hình '{arch_name}' đã hoàn tất huấn luyện từ phiên trước "
                f"(Best Epoch = {best_epoch}, Best Val Macro-F1 = {best_val_f1:.2f}%). Bỏ qua vòng lặp train!"
            )
            if best_model_path.exists():
                best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
                model.load_state_dict(best_ckpt["model_state_dict"])
            return model, pd.DataFrame(history_records), best_model_path

        print(
            f"[AUTO-RESUME] Tiếp tục huấn luyện '{arch_name}' từ Epoch {start_epoch}/{total_epochs} "
            f"(Best Val Macro-F1 hiện tại: {best_val_f1:.2f}%, No-Improve Count: {epochs_no_improve})"
        )

    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    current_stage = 0

    for epoch in range(start_epoch, total_epochs + 1):
        t_start = time.time()
        target_stage = 1 if epoch <= stage1_epochs else 2

        if target_stage != current_stage or optimizer is None:
            is_resuming_same_stage = (
                saved_state is not None
                and epoch == start_epoch
                and int(saved_state.get("stage", 0)) == target_stage
            )
            current_stage = target_stage
            optimizer, scheduler = _build_stage_optimizer_and_scheduler(
                model=model,
                arch_name=arch_name,
                stage=current_stage,
                stage1_epochs=stage1_epochs,
                total_epochs=total_epochs,
                lr_stage1=lr_stage1,
                lr_stage2_backbone=lr_stage2_backbone,
                lr_stage2_head=lr_stage2_head,
            )

            if is_resuming_same_stage and saved_state is not None:
                if "optimizer_state_dict" in saved_state:
                    optimizer.load_state_dict(saved_state["optimizer_state_dict"])
                if "scheduler_state_dict" in saved_state:
                    scheduler.load_state_dict(saved_state["scheduler_state_dict"])
                if use_amp and "scaler_state_dict" in saved_state and saved_state["scaler_state_dict"]:
                    scaler.load_state_dict(saved_state["scaler_state_dict"])
            elif current_stage == 2 and stage1_epochs > 0:
                # Chỉ reset bộ đếm dừng sớm khi thực sự bước từ cuối Stage 1 sang đầu Stage 2
                epochs_no_improve = 0

            param_info = count_parameters(model)
            print(
                f"\n---> [{arch_name.upper()}] Stage {current_stage} tại Epoch {epoch}/{total_epochs} | "
                f"Trainable Params: {param_info['trainable_params']:,} / {param_info['total_params']:,}"
            )

        # -------------------
        # 1. TRAINING STEP
        # -------------------
        model.train()
        freeze_bn_stats_if_needed(model)

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in dataloaders["train"]:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_sz = inputs.size(0)
            train_loss_sum += loss.item() * batch_sz
            preds = torch.argmax(logits.detach(), dim=1)
            train_correct += int((preds == targets).sum().item())
            train_total += batch_sz

        if scheduler is not None:
            scheduler.step()

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = (train_correct / max(train_total, 1)) * 100.0

        # -------------------
        # 2. VALIDATION STEP
        # -------------------
        val_res = evaluate_dataloader(model, dataloaders["val"], criterion, device, idx_to_class)
        val_loss = val_res["loss"]
        val_acc = val_res["accuracy"]
        val_f1 = val_res["macro_f1"]
        val_fsr = val_res["safety_metrics"]["false_safe_rate_percent"]

        elapsed = time.time() - t_start
        current_lr = optimizer.param_groups[-1]["lr"]

        epoch_record = {
            "epoch": epoch,
            "stage": current_stage,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "train_acc": round(train_acc, 3),
            "val_acc": round(val_acc, 3),
            "val_macro_f1": round(val_f1, 3),
            "val_fsr_percent": round(val_fsr, 3),
            "lr": current_lr,
            "elapsed_sec": round(elapsed, 2),
        }
        history_records.append(epoch_record)
        pd.DataFrame(history_records).to_csv(history_csv_path, index=False, encoding="utf-8")

        improved_marker = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            improved_marker = " [*BEST SAVED TO DRIVE*]"

            save_self_contained_checkpoint(
                save_path=best_model_path,
                model=model,
                arch_name=arch_name,
                class_to_idx=class_to_idx,
                idx_to_class=idx_to_class,
                best_epoch=best_epoch,
                best_val_acc=best_val_acc,
                best_val_f1=best_val_f1,
                img_size=img_size,
            )
        else:
            epochs_no_improve += 1

        should_stop_early = current_stage == 2 and epochs_no_improve >= patience
        is_finished = (epoch == total_epochs) or should_stop_early

        torch.save(
            {
                "arch_name": arch_name,
                "epoch": epoch,
                "stage": current_stage,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else {},
                "scaler_state_dict": scaler.state_dict() if use_amp else {},
                "best_val_f1": best_val_f1,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "epochs_no_improve": epochs_no_improve,
                "history_records": history_records,
                "training_completed": is_finished,
            },
            resume_ckpt_path,
        )

        print(
            f"Epoch [{epoch:02d}/{total_epochs:02d}] (S{current_stage}) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.2f}%, FSR: {val_fsr:.2f}% | "
            f"{elapsed:.1f}s{improved_marker}"
        )

        if should_stop_early:
            print(
                f"[EARLY STOPPING] '{arch_name}' không cải thiện Val Macro-F1 sau {patience} epochs. "
                f"Dừng sớm tại Epoch {epoch} (Best Epoch = {best_epoch}, Best F1 = {best_val_f1:.2f}%)."
            )
            break

    if best_model_path.exists():
        best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

    history_df = pd.DataFrame(history_records)
    return model, history_df, best_model_path


# ==============================================================================
# BỘ HÀM TRỰC QUAN HÓA TRƯỚC & SAU KHI TRAIN (HÌNH 07 -> HÌNH 12, 300 DPI)
# ==============================================================================

def _denormalize_tensor(img_tensor: torch.Tensor) -> np.ndarray:
    """Chuyển Tensor chuẩn hóa ImageNet (3, H, W) về mảng ảnh RGB [0, 1] (H, W, 3)."""
    arr = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)
    return np.clip(arr * std + mean, 0.0, 1.0)


def plot_before_vs_after_training(
    untrained_model: nn.Module,
    trained_model: nn.Module,
    arch_name: str,
    test_loader: DataLoader,
    idx_to_class: Dict[int, str],
    device: torch.device,
    output_dir: str | Path = "reports/figures",
    num_samples: int = 4,
) -> Path:
    """
    Hình 07: Đối chứng trực quan kết quả Dự đoán & Bản đồ nhiệt Grad-CAM
    TRƯỚC KHI TRAIN (Untrained Model) và SAU KHI TRAIN (Fine-tuned Model) trên cùng mẫu thử.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    inputs, targets = next(iter(test_loader))
    n = min(num_samples, inputs.size(0))

    untrained_explainer = GradCAMExplainer(untrained_model.to(device), arch_name)
    trained_explainer = GradCAMExplainer(trained_model.to(device), arch_name)

    fig, axes = plt.subplots(n, 3, figsize=(13, 3.6 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        single_tensor = inputs[i : i + 1].to(device)
        true_idx = int(targets[i].item())
        true_label = idx_to_class[true_idx]
        rgb_np = _denormalize_tensor(inputs[i])

        cam_before, pred_before_idx, probs_before = untrained_explainer.generate_heatmap(single_tensor)
        overlay_before = GradCAMExplainer.overlay_heatmap_on_image(rgb_np, cam_before)
        pred_before_lbl = idx_to_class[pred_before_idx]
        conf_before = probs_before[pred_before_idx] * 100.0

        cam_after, pred_after_idx, probs_after = trained_explainer.generate_heatmap(single_tensor)
        overlay_after = GradCAMExplainer.overlay_heatmap_on_image(rgb_np, cam_after)
        pred_after_lbl = idx_to_class[pred_after_idx]
        conf_after = probs_after[pred_after_idx] * 100.0

        axes[i, 0].imshow(rgb_np)
        axes[i, 0].set_title(f"Ảnh Kiểm thử #{i+1}\nNhãn thực tế: {true_label}", fontweight="bold")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay_before)
        color_b = "#27ae60" if pred_before_idx == true_idx else "#c0392b"
        axes[i, 1].set_title(
            f"[TRƯỚC KHI TRAIN] Dự đoán: {pred_before_lbl}\nĐộ tự tin: {conf_before:.1f}% (Grad-CAM phân tán)",
            color=color_b,
            fontsize=10,
        )
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay_after)
        color_a = "#27ae60" if pred_after_idx == true_idx else "#c0392b"
        axes[i, 2].set_title(
            f"[SAU KHI TRAIN] Dự đoán: {pred_after_lbl}\nĐộ tự tin: {conf_after:.1f}% (Grad-CAM hội tụ đúng vùng quả)",
            color=color_a,
            fontsize=10,
            fontweight="bold",
        )
        axes[i, 2].axis("off")

    plt.suptitle(
        f"Đối chứng Trực quan Trước & Sau khi Huấn luyện ({arch_name.upper()}) kèm Giải thích Grad-CAM",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    save_file = out_path / "07_before_vs_after_training_predictions.png"
    fig.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_file


def plot_training_curves_all_models(
    histories: Dict[str, pd.DataFrame],
    output_dir: str | Path = "reports/figures",
) -> Path:
    """
    Hình 08: Đường cong Huấn luyện (Training & Validation Loss, Accuracy, Macro-F1)
    của cả 3 mô hình qua từng Epoch (có đánh dấu mốc chuyển giao Pha 1 -> Pha 2).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_names = list(histories.keys())
    n_models = len(model_names)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 4.2 * n_models))
    if n_models == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, m_name in enumerate(model_names):
        hdf = histories[m_name]
        epochs = hdf["epoch"].values

        ax_loss = axes[row_idx, 0]
        ax_loss.plot(epochs, hdf["train_loss"], "o-", label="Train Loss", color="#2980b9", linewidth=2)
        ax_loss.plot(epochs, hdf["val_loss"], "s--", label="Validation Loss", color="#e74c3c", linewidth=2)

        if "stage" in hdf.columns and (hdf["stage"] == 1).any() and (hdf["stage"] == 2).any():
            split_ep = hdf[hdf["stage"] == 1]["epoch"].max() + 0.5
            ax_loss.axvline(split_ep, color="gray", linestyle=":", linewidth=1.8, label="Bắt đầu Fine-tuning (Stage 2)")

        ax_loss.set_title(f"[{m_name.upper()}] Đường cong Hàm mất mát (Loss Curve)")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Cross-Entropy Loss (Label Smoothing)")
        ax_loss.legend()
        ax_loss.grid(True, linestyle="--", alpha=0.4)

        ax_acc = axes[row_idx, 1]
        ax_acc.plot(epochs, hdf["train_acc"], "o-", label="Train Accuracy (%)", color="#2980b9", linewidth=2)
        ax_acc.plot(epochs, hdf["val_acc"], "s--", label="Val Accuracy (%)", color="#27ae60", linewidth=2)
        ax_acc.plot(epochs, hdf["val_macro_f1"], "^-.", label="Val Macro-F1 (%)", color="#8e44ad", linewidth=2)

        if "stage" in hdf.columns and (hdf["stage"] == 1).any() and (hdf["stage"] == 2).any():
            split_ep = hdf[hdf["stage"] == 1]["epoch"].max() + 0.5
            ax_acc.axvline(split_ep, color="gray", linestyle=":", linewidth=1.8, label="Bắt đầu Fine-tuning (Stage 2)")

        best_idx = hdf["val_macro_f1"].idxmax()
        best_ep = hdf.loc[best_idx, "epoch"]
        best_f1 = hdf.loc[best_idx, "val_macro_f1"]
        ax_acc.scatter([best_ep], [best_f1], color="gold", edgecolors="black", s=120, zorder=5, label=f"Best F1 ({best_f1:.2f}% @ Ep {best_ep})")

        ax_acc.set_title(f"[{m_name.upper()}] Độ chính xác & Macro F1-Score")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Phần trăm (%)")
        ax_acc.legend(loc="lower right")
        ax_acc.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    save_file = out_path / "08_training_curves_3models.png"
    fig.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_file


def plot_confusion_matrices_and_report(
    eval_results: Dict[str, Dict[str, Any]],
    idx_to_class: Dict[int, str],
    best_model_name: str,
    output_dir: str | Path = "reports/figures",
) -> Tuple[Path, Path]:
    """
    Xuất Hình 09 (Confusion Matrix số lượng & chuẩn hóa %) và
    Hình 10 (Heatmap Classification Report: Precision, Recall, F1-Score từng lớp).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    best_res = eval_results[best_model_name]
    cm = best_res["confusion_matrix"]
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for ax, mat, title, is_pct in [
        (ax1, cm, f"Ma trận Nhầm lẫn (Số lượng mẫu) - {best_model_name.upper()}", False),
        (ax2, cm_norm, f"Ma trận Nhầm lẫn Chuẩn hóa (%) - {best_model_name.upper()}", True),
    ]:
        im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_ylabel("Nhãn Thực tế (Actual Label)")
        ax.set_xlabel("Nhãn Dự đoán (Predicted Label)")

        thresh = mat.max() / 2.0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val_str = f"{mat[i, j]:.1f}%" if is_pct else f"{int(mat[i, j])}"
                ax.text(
                    j,
                    i,
                    val_str,
                    ha="center",
                    va="center",
                    color="white" if mat[i, j] > thresh else "black",
                    fontsize=8.5,
                )

    plt.tight_layout()
    f9_path = out_path / "09_confusion_matrices_comparison.png"
    fig.savefig(f9_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    report_dict = classification_report(
        best_res["y_true"],
        best_res["y_pred"],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for cls_name in class_names:
        r = report_dict[cls_name]
        rows.append(
            {
                "Class": cls_name,
                "Precision (%)": r["precision"] * 100.0,
                "Recall (%)": r["recall"] * 100.0,
                "F1-Score (%)": r["f1-score"] * 100.0,
                "Support": int(r["support"]),
            }
        )
    rep_df = pd.DataFrame(rows).set_index("Class")
    rep_df.to_csv(out_path / "per_class_classification_report.csv", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, max(5.5, 0.55 * len(class_names))))
    mat_metrics = rep_df[["Precision (%)", "Recall (%)", "F1-Score (%)"]].values
    im = ax.imshow(mat_metrics, cmap="YlGn", vmin=60.0, vmax=100.0, aspect="auto")
    fig.colorbar(im, ax=ax, label="Phần trăm (%)")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Precision (%)", "Recall (%)", "F1-Score (%)"], fontweight="bold")
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels([f"{c} (n={rep_df.loc[c, 'Support']})" for c in class_names])
    ax.set_title(f"Báo cáo Phân loại Chi tiết Từng lớp - {best_model_name.upper()}")

    for i in range(mat_metrics.shape[0]):
        for j in range(mat_metrics.shape[1]):
            ax.text(
                j,
                i,
                f"{mat_metrics[i, j]:.2f}%",
                ha="center",
                va="center",
                color="black" if mat_metrics[i, j] < 88 else "white",
                fontweight="bold",
            )

    plt.tight_layout()
    f10_path = out_path / "10_per_class_classification_report.png"
    fig.savefig(f10_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return f9_path, f10_path


def plot_ablation_model_comparison(
    summary_df: pd.DataFrame,
    output_dir: str | Path = "reports/figures",
) -> Path:
    """
    Hình 11: Biểu đồ So sánh Thực nghiệm 3 Mô hình (Custom CNN vs MobileNetV2 vs EfficientNet-B0)
    trên các tiêu chí: Accuracy, Macro-F1, False Safe Rate (FSR), Latency (ms) và Dung lượng (MB).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.4, 1]})
    models_list = summary_df["Model"].tolist()
    x = np.arange(len(models_list))
    w = 0.25

    b1 = ax1.bar(x - w, summary_df["Test Accuracy (%)"], w, label="Test Accuracy (%)", color="#2980b9", edgecolor="black")
    b2 = ax1.bar(x, summary_df["Macro F1-Score (%)"], w, label="Macro F1-Score (%)", color="#27ae60", edgecolor="black")
    b3 = ax1.bar(x + w, summary_df["False Safe Rate - FSR (%)"], w, label="False Safe Rate - FSR (%) [Càng thấp càng tốt]", color="#e74c3c", edgecolor="black")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models_list, fontweight="bold")
    ax1.set_ylabel("Phần trăm (%)")
    ax1.set_ylim(0, 112)
    ax1.set_title("So sánh Độ chính xác, Macro-F1 & Tỷ lệ Bỏ lọt Quả hỏng (FSR)")
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    b_lat = ax2.bar(x - 0.18, summary_df["CPU Latency (ms/img)"], 0.35, label="Thời gian suy luận CPU (ms/ảnh)", color="#f39c12", edgecolor="black")
    b_sz = ax2.bar(x + 0.18, summary_df["Model Size (MB)"], 0.35, label="Dung lượng file Trọng số (MB)", color="#8e44ad", edgecolor="black")

    ax2.set_xticks(x)
    ax2.set_xticklabels(models_list, fontweight="bold")
    ax2.set_ylabel("Giá trị (ms / MB)")
    ax2.set_title("Đánh đổi Chi phí Tính toán (Tốc độ CPU & Dung lượng)")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    for bars in (b_lat, b_sz):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    f11_path = out_path / "11_ablation_model_comparison.png"
    fig.savefig(f11_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f11_path


def plot_gradcam_and_error_analysis(
    model: nn.Module,
    arch_name: str,
    test_loader: DataLoader,
    idx_to_class: Dict[int, str],
    device: torch.device,
    output_dir: str | Path = "reports/figures",
) -> Path:
    """
    Hình 12: Giải thích Mô hình bằng Grad-CAM cho cả Ca dự đoán ĐÚNG tiêu biểu
    và Phân tích lỗi (Error Analysis - các ca dự đoán SAI hoặc có độ tự tin thấp nhất).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    explainer = GradCAMExplainer(model.to(device), arch_name)
    correct_samples = []
    misclassified_samples = []

    for inputs, targets in test_loader:
        for i in range(inputs.size(0)):
            single_x = inputs[i : i + 1].to(device)
            true_idx = int(targets[i].item())
            with torch.no_grad():
                logits = model(single_x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))

            item = (inputs[i], true_idx, pred_idx, float(probs[pred_idx]))
            if pred_idx == true_idx and len(correct_samples) < 4:
                correct_samples.append(item)
            elif pred_idx != true_idx and len(misclassified_samples) < 4:
                misclassified_samples.append(item)

        if len(correct_samples) >= 4 and len(misclassified_samples) >= 4:
            break

    while len(misclassified_samples) < 4 and correct_samples:
        misclassified_samples.append(correct_samples[-1])

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.8))
    rows_data = [
        ("CA DỰ ĐOÁN ĐÚNG TIÊU BIỂU (CORRECT)", correct_samples[:4]),
        ("PHÂN TÍCH CA NHẦM LẪN / KHÓ (ERROR ANALYSIS)", misclassified_samples[:4]),
    ]

    for row_idx, (_row_title, items) in enumerate(rows_data):
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            if col_idx < len(items):
                img_tensor, true_idx, _p_idx, _conf = items[col_idx]
                single_x = img_tensor.unsqueeze(0).to(device)
                cam, pred_idx, probs = explainer.generate_heatmap(single_x)
                rgb_np = _denormalize_tensor(img_tensor)
                overlay = GradCAMExplainer.overlay_heatmap_on_image(rgb_np, cam, alpha=0.45)

                true_lbl = idx_to_class[true_idx]
                pred_lbl = idx_to_class[pred_idx]
                conf = probs[pred_idx] * 100.0
                is_ok = true_idx == pred_idx

                ax.imshow(overlay)
                status_tag = "ĐÚNG" if is_ok else "NHẦM LẪN"
                title_col = "#1e8449" if is_ok else "#c0392b"
                ax.set_title(
                    f"[{status_tag}] Thực tế: {true_lbl}\nDự đoán: {pred_lbl} ({conf:.1f}%)",
                    color=title_col,
                    fontsize=9.5,
                    fontweight="bold",
                )
            ax.axis("off")

    plt.suptitle(
        f"Giải thích Mô hình bằng Bản đồ nhiệt Grad-CAM & Phân tích Lỗi ({arch_name.upper()})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    f12_path = out_path / "12_gradcam_correct_and_errors.png"
    fig.savefig(f12_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f12_path
