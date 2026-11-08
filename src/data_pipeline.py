"""
Giai đoạn 2 (v2.0 - Đã hiệu đính theo Kiểm định Chéo):
Module Quản lý Dataset, Làm sạch dữ liệu, Chống Data Leakage (dHash),
Phân tích Khám phá Dữ liệu (EDA - Foreground Masking) và Pipeline Tiền xử lý PyTorch.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# Cho phép đọc các file ảnh bị cắt nhẹ ở cuối nếu vẫn hợp lệ
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Chuẩn hóa ImageNet tiêu chuẩn cho MobileNetV2, EfficientNet-B0 và Custom CNN
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_class_name(raw_folder_name: str) -> Optional[str]:
    """
    Chuẩn hóa tên thư mục từ các bộ dữ liệu Kaggle khác nhau (cả tiền tố lẫn hậu tố)
    về định dạng thống nhất: `fresh_<fruit>` hoặc `rotten_<fruit>`.
    Ví dụ:
      - 'freshapples' / 'Fresh_Apple' / 'apple_fresh' -> 'fresh_apple'
      - 'rottenbanana' / 'stale_banana' / 'banana_rotten' -> 'rotten_banana'
    """
    cleaned = raw_folder_name.strip().lower()
    cleaned = re.sub(r"[\s\-]+", "_", cleaned)

    fresh_keywords = ("fresh", "good", "healthy")
    rotten_keywords = ("rotten", "stale", "spoiled", "bad", "decayed")

    status = None
    fruit_part = ""

    # 1. Kiểm tra dạng tiền tố (fresh_apple, freshapples, rotten_banana...)
    for prefix in fresh_keywords:
        if cleaned.startswith(prefix):
            status = "fresh"
            fruit_part = cleaned[len(prefix) :].strip("_")
            break
    if status is None:
        for prefix in rotten_keywords:
            if cleaned.startswith(prefix):
                status = "rotten"
                fruit_part = cleaned[len(prefix) :].strip("_")
                break

    # 2. Kiểm tra dạng hậu tố (apple_fresh, banana_rotten, orange_stale...)
    if status is None or not fruit_part:
        for suffix in fresh_keywords:
            if cleaned.endswith(suffix):
                status = "fresh"
                fruit_part = cleaned[: -len(suffix)].strip("_")
                break
    if status is None or not fruit_part:
        for suffix in rotten_keywords:
            if cleaned.endswith(suffix):
                status = "rotten"
                fruit_part = cleaned[: -len(suffix)].strip("_")
                break

    if status is None or not fruit_part:
        return None

    # Chuẩn hóa số nhiều về số ít phổ biến
    plural_map = {
        "apples": "apple",
        "bananas": "banana",
        "oranges": "orange",
        "mangoes": "mango",
        "mangos": "mango",
        "strawberries": "strawberry",
        "peaches": "peach",
        "tomatoes": "tomato",
        "potatoes": "potato",
        "cucumbers": "cucumber",
        "capsicums": "capsicum",
        "bittergourds": "bittergourd",
        "bitter_gourds": "bittergourd",
        "bitter_gourd": "bittergourd",
        "grapes": "grape",
        "guavas": "guava",
        "pomegranates": "pomegranate",
    }
    fruit_part = plural_map.get(fruit_part, fruit_part)
    return f"{status}_{fruit_part}"


def load_rgb_image_safe(image_path: str | Path) -> Image.Image:
    """
    Đọc ảnh an toàn và xử lý triệt để kênh trong suốt (RGBA / Palette 'P')
    bằng cách ghép lên nền trắng (255, 255, 255) trước khi chuyển sang RGB.
    """
    with Image.open(image_path) as img:
        img.load()
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            composited = Image.alpha_composite(white_bg, rgba)
            return composited.convert("RGB")
        return img.convert("RGB")


def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """
    Tính Difference Hash (dHash) 64-bit để phát hiện ảnh trùng lặp (Data Leakage)
    giữa các thư mục train/test gốc của bộ dữ liệu Kaggle.
    """
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bit_string = "".join("1" if b else "0" for b in diff.flatten())
    return f"{int(bit_string, 2):016x}"


def download_kaggle_dataset(
    dataset_slug: str = "sriramr/fruits-fresh-and-rotten-for-classification",
) -> Path:
    """
    Tự động tải bộ dữ liệu từ Kaggle thông qua `kagglehub`.
    Mặc định hỗ trợ:
      - 'sriramr/fruits-fresh-and-rotten-for-classification' (6 lớp chuẩn)
      - 'swoyam2609/fresh-and-stale-classification' (12 lớp mở rộng)
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "Vui lòng cài đặt kagglehub: `pip install kagglehub`"
        ) from exc

    print(f"[INFO] Đang tải bộ dữ liệu Kaggle: {dataset_slug} ...")
    downloaded_path = Path(kagglehub.dataset_download(dataset_slug))
    print(f"[INFO] Đã tải thành công tại: {downloaded_path}")
    return downloaded_path


def _is_redundant_nested_folder(file_path: Path, root_path: Path) -> bool:
    """
    Phát hiện nhánh thư mục bị đóng gói đúp kinh điển trên Kaggle
    (ví dụ: `dataset/dataset/train/...` trong bộ `sriramr/fruits-fresh-and-rotten`).
    """
    rel_parts = [p.lower() for p in file_path.relative_to(root_path).parts[:-1]]
    for i in range(len(rel_parts) - 1):
        if rel_parts[i] == rel_parts[i + 1] and rel_parts[i] in {"dataset", "data", "fruits", "images"}:
            return True
    return False


def scan_and_clean_dataset(
    root_dir: str | Path,
    allowed_fruits: Optional[List[str]] = None,
    min_dimension: int = 32,
    deduplicate: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Quét toàn bộ thư mục dữ liệu, chặn nhánh đóng gói đúp (`dataset/dataset/`),
    chuẩn hóa tên lớp (hỗ trợ cả 1 cấp và 2 cấp thư mục), loại bỏ ảnh hỏng,
    và khử trùng lặp bằng dHash để chống Data Leakage giữa Train & Test.
    """
    root_path = Path(root_dir)
    records = []
    seen_hashes: Dict[str, str] = {}
    stats = {
        "nested_redundant_skipped": 0,
        "total_scanned": 0,
        "corrupted_or_too_small": 0,
        "duplicates_removed": 0,
        "valid_kept": 0,
    }

    for file_path in sorted(root_path.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        # Chặn nhánh thư mục đóng gói đúp `dataset/dataset/` để không làm lệch thống kê
        if _is_redundant_nested_folder(file_path, root_path):
            stats["nested_redundant_skipped"] += 1
            continue

        parent_name = file_path.parent.name
        norm_label = normalize_class_name(parent_name)

        # Hỗ trợ cấu trúc thư mục 2 cấp (ví dụ: Fresh/Apple/img.jpg hoặc Apple/Fresh/img.jpg)
        if norm_label is None and file_path.parent != root_path:
            grandparent_name = file_path.parent.parent.name
            norm_label = normalize_class_name(f"{grandparent_name}_{parent_name}")

        if norm_label is None:
            continue

        status, fruit_type = norm_label.split("_", 1)
        if allowed_fruits is not None and fruit_type not in allowed_fruits:
            continue

        stats["total_scanned"] += 1

        try:
            img = load_rgb_image_safe(file_path)
            width, height = img.size
            if width < min_dimension or height < min_dimension:
                stats["corrupted_or_too_small"] += 1
                continue

            img_hash = compute_dhash(img)
            if deduplicate and img_hash in seen_hashes:
                stats["duplicates_removed"] += 1
                continue

            seen_hashes[img_hash] = str(file_path)
            stats["valid_kept"] += 1

            records.append(
                {
                    "filepath": str(file_path.resolve()),
                    "label": norm_label,
                    "status": status,       # 'fresh' hoặc 'rotten'
                    "fruit_type": fruit_type,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 4),
                    "dhash": img_hash,
                }
            )
        except Exception:
            stats["corrupted_or_too_small"] += 1

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(
            f"Không tìm thấy ảnh hợp lệ trong {root_dir}. Hãy kiểm tra lại cấu trúc thư mục."
        )

    print(
        f"[CLEANING SUMMARY] Đã quét: {stats['total_scanned']} | "
        f"Bỏ qua nhánh lặp đóng gói: {stats['nested_redundant_skipped']} | "
        f"Ảnh lỗi/nhỏ: {stats['corrupted_or_too_small']} | "
        f"Ảnh trùng lặp (dHash): {stats['duplicates_removed']} | "
        f"Ảnh hợp lệ giữ lại: {stats['valid_kept']}"
    )
    return df, stats


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    save_manifest_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Chia tập dữ liệu theo phương pháp Stratified Split (70% Train - 15% Val - 15% Test)
    đảm bảo phân phối các lớp đồng nhất tuyệt đối trên cả 3 tập.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Tổng tỷ lệ phải bằng 1.0"

    df = df.copy().reset_index(drop=True)
    splitter_1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1.0 - train_ratio), random_state=seed
    )
    train_idx, temp_idx = next(splitter_1.split(df, df["label"]))

    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    splitter_2 = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_test_ratio, random_state=seed
    )
    val_sub_idx, test_sub_idx = next(splitter_2.split(temp_df, temp_df["label"]))

    df["split"] = "train"
    df.loc[temp_idx[val_sub_idx], "split"] = "val"
    df.loc[temp_idx[test_sub_idx], "split"] = "test"

    if save_manifest_path is not None:
        save_path = Path(save_manifest_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"[INFO] Đã lưu manifest phân chia dữ liệu tại: {save_path}")

    return df


def get_transforms(img_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Thiết kế Pipeline Tiền xử lý & Tăng cường dữ liệu (Data Augmentation)
    bảo toàn ngữ nghĩa màu sắc chất lượng hoa quả (Label-Preserving Augmentation).
    Lưu ý: Đặt `RandomRotation(degrees=20, fill=(255, 255, 255))` để tránh tạo góc đen
    trên ảnh nền trắng làm nhiễu đặc trưng quả hỏng.
    """
    resize_dim = int(img_size * 256 / 224)
    train_transform = transforms.Compose(
        [
            transforms.Resize((resize_dim, resize_dim)),
            transforms.RandomRotation(degrees=20, fill=(255, 255, 255)),
            transforms.RandomResizedCrop(img_size, scale=(0.80, 1.0), ratio=(0.90, 1.10)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            # Giữ hue=0.02 rất nhỏ để không làm quả tươi bị đổi màu thành quả thối
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), value="random"),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
    }


class FruitFreshnessDataset(Dataset):
    """PyTorch Dataset tùy chỉnh đọc từ DataFrame manifest đã làm sạch."""

    def __init__(
        self,
        df: pd.DataFrame,
        class_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.filepaths = self.df["filepath"].tolist()
        self.labels = [self.class_to_idx[lbl] for lbl in self.df["label"].tolist()]

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = load_rgb_image_safe(self.filepaths[idx])
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def compute_class_weights(
    train_df: pd.DataFrame, class_to_idx: Dict[str, int], device: torch.device
) -> torch.Tensor:
    """
    Tính trọng số lớp cân bằng w_c = N_total / (C * n_c) để đưa vào CrossEntropyLoss.
    Lưu ý Học thuật: Không bật đồng thời `class_weights` trong Loss và `WeightedRandomSampler`
    trong DataLoader để tránh hiện tượng phạt kép bậc hai (Quadratic Over-weighting).
    """
    counts = Counter(train_df["label"].tolist())
    num_classes = len(class_to_idx)
    total_samples = len(train_df)

    weights = np.zeros(num_classes, dtype=np.float32)
    for cls_name, idx in class_to_idx.items():
        n_c = max(counts.get(cls_name, 1), 1)
        weights[idx] = total_samples / (num_classes * n_c)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_dataloaders(
    df: pd.DataFrame,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    use_weighted_sampler: bool = False,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], Dict[int, str]]:
    """
    Khởi tạo bộ DataLoader cho Train, Val, Test kèm từ điển ánh xạ nhãn chuẩn.
    Mặc định `use_weighted_sampler=False` (dùng `shuffle=True` kết hợp `class_weights` trong Loss).
    Nếu bật `use_weighted_sampler=True`, hãy đặt `weight=None` trong `CrossEntropyLoss`.
    """
    class_names = sorted(df["label"].unique().tolist())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(class_names)}

    tfms = get_transforms(img_size=img_size)
    dataloaders: Dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        sub_df = df[df["split"] == split].reset_index(drop=True)
        ds = FruitFreshnessDataset(
            df=sub_df, class_to_idx=class_to_idx, transform=tfms[split]
        )

        sampler = None
        shuffle = split == "train"

        if split == "train" and use_weighted_sampler:
            label_counts = Counter(ds.labels)
            sample_weights = [1.0 / label_counts[lbl] for lbl in ds.labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights), replacement=True
            )
            shuffle = False

        dataloaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return dataloaders, class_to_idx, idx_to_class


# ==============================================================================
# BỘ HÀM XUẤT 6 BIỂU ĐỒ EDA TRƯỚC KHI TRAIN (300 DPI CHO BÁO CÁO)
# ==============================================================================

def generate_all_eda_figures(
    df: pd.DataFrame,
    output_dir: str | Path = "reports/figures",
    img_size: int = 224,
) -> List[Path]:
    """
    Tự động vẽ và lưu trọn bộ 6 biểu đồ EDA chất lượng 300 DPI phục vụ Báo cáo học thuật.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []

    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"

    # ------------------------------------------------------------------
    # Hình 1: Phân phối số lượng mẫu theo từng lớp & Tỷ lệ Fresh vs Rotten
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.8, 1]})
    class_counts = df["label"].value_counts().sort_index()
    colors = ["#2ecc71" if lbl.startswith("fresh_") else "#e74c3c" for lbl in class_counts.index]

    bars = ax1.barh(class_counts.index, class_counts.values, color=colors, edgecolor="black", alpha=0.88)
    ax1.set_title("Số lượng ảnh theo từng lớp (Xanh: Tươi | Đỏ: Hỏng)")
    ax1.set_xlabel("Số lượng ảnh")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + max(class_counts.values) * 0.01, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", fontsize=10)
    ax1.set_xlim(0, max(class_counts.values) * 1.15)

    status_counts = df["status"].value_counts()
    ax2.pie(
        status_counts.values,
        labels=[f"{s.upper()}\n({c} ảnh)" for s, c in status_counts.items()],
        autopct="%1.1f%%",
        startangle=140,
        colors=["#2ecc71" if s == "fresh" else "#e74c3c" for s in status_counts.index],
        wedgeprops={"edgecolor": "black", "linewidth": 1},
        explode=[0.03] * len(status_counts),
    )
    imbalance_ratio = class_counts.max() / max(class_counts.min(), 1)
    ax2.set_title(f"Tỷ lệ Fresh vs. Rotten\n(Imbalance Ratio = {imbalance_ratio:.2f}x)")

    plt.tight_layout()
    f1_path = out_path / "01_class_distribution.png"
    fig.savefig(f1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(f1_path)

    # ------------------------------------------------------------------
    # Hình 2: Kiểm chứng phân phối Stratified Split (Train / Val / Test)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))
    split_pivot = df.pivot_table(index="label", columns="split", values="filepath", aggfunc="count", fill_value=0)
    for col in ("train", "val", "test"):
        if col not in split_pivot.columns:
            split_pivot[col] = 0
    split_pivot = split_pivot[["train", "val", "test"]]

    split_pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#3498db", "#f39c12", "#9b59b6"],
        edgecolor="black",
        alpha=0.9,
    )
    ax.set_title("Phân bố mẫu sau khi chia Stratified Split (70% Train - 15% Val - 15% Test)")
    ax.set_xlabel("Tên lớp")
    ax.set_ylabel("Số lượng mẫu")
    ax.legend(["Train (70%)", "Validation (15%)", "Test (15%)"], loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    f2_path = out_path / "02_stratified_split_verification.png"
    fig.savefig(f2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(f2_path)

    # ------------------------------------------------------------------
    # Hình 3: Phân tích kích thước gốc (Width x Height) & Aspect Ratio
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sample_dim_df = df.sample(n=min(2000, len(df)), random_state=42)
    ax1.scatter(
        sample_dim_df["width"],
        sample_dim_df["height"],
        c=["#2ecc71" if s == "fresh" else "#e74c3c" for s in sample_dim_df["status"]],
        alpha=0.5,
        edgecolors="none",
        s=25,
    )
    ax1.axvline(img_size, color="navy", linestyle="--", label=f"Target Size ({img_size}x{img_size})")
    ax1.axhline(img_size, color="navy", linestyle="--")
    ax1.set_title("Phân tán Kích thước ảnh gốc (Width vs. Height)")
    ax1.set_xlabel("Chiều rộng (Width - px)")
    ax1.set_ylabel("Chiều cao (Height - px)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.hist(df["aspect_ratio"], bins=35, color="#34495e", edgecolor="white", alpha=0.85)
    ax2.axvline(1.0, color="#e74c3c", linestyle="--", linewidth=2, label="Tỷ lệ chuẩn 1:1 (AR = 1.0)")
    ax2.set_title("Phân phối Tỷ lệ Khung hình (Aspect Ratio = W / H)")
    ax2.set_xlabel("Aspect Ratio (W / H)")
    ax2.set_ylabel("Số lượng ảnh")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    f3_path = out_path / "03_image_dimensions_analysis.png"
    fig.savefig(f3_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(f3_path)

    # ------------------------------------------------------------------
    # Hình 4: Lưới ảnh mẫu đối chứng Tươi (Fresh) vs. Hỏng (Rotten)
    # ------------------------------------------------------------------
    fruits = sorted(df["fruit_type"].unique().tolist())
    n_cols = len(fruits)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.5))
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for col_idx, fruit in enumerate(fruits):
        for row_idx, status in enumerate(["fresh", "rotten"]):
            ax = axes[row_idx, col_idx]
            target_label = f"{status}_{fruit}"
            sub = df[df["label"] == target_label]
            if not sub.empty:
                row = sub.iloc[0]
                img = load_rgb_image_safe(row["filepath"]).resize((224, 224))
                ax.imshow(img)
                title_color = "#1e8449" if status == "fresh" else "#c0392b"
                ax.set_title(f"{target_label}\n({row['width']}x{row['height']}px)", color=title_color, fontsize=10)
            ax.axis("off")

    plt.suptitle("Đối chứng Mẫu ảnh gốc giữa Trạng thái Tươi (Hàng trên) và Hỏng (Hàng dưới)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    f4_path = out_path / "04_raw_samples_comparison.png"
    fig.savefig(f4_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(f4_path)

    # ------------------------------------------------------------------
    # Hình 5: Phân tích phổ màu RGB & Độ sáng trên Vùng Tiền Cảnh Quả
    # (Loại bỏ pixel nền trắng R>0.94 & G>0.94 & B>0.94 để tránh nhiễu nền)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fresh_brightness, rotten_brightness = [], []
    fresh_rgb_means = {"R": [], "G": [], "B": []}
    rotten_rgb_means = {"R": [], "G": [], "B": []}

    for status, b_list, rgb_dict in [
        ("fresh", fresh_brightness, fresh_rgb_means),
        ("rotten", rotten_brightness, rotten_rgb_means),
    ]:
        sub_df = df[df["status"] == status]
        sampled = sub_df.sample(n=min(200, len(sub_df)), random_state=42)
        for fp in sampled["filepath"]:
            arr = np.asarray(load_rgb_image_safe(fp).resize((128, 128)), dtype=np.float32) / 255.0
            # Tách pixel tiền cảnh của quả (Foreground Masking: loại bỏ nền trắng studio)
            fg_mask = ~((arr[:, :, 0] > 0.94) & (arr[:, :, 1] > 0.94) & (arr[:, :, 2] > 0.94))
            if np.any(fg_mask):
                fg_pixels = arr[fg_mask]
            else:
                fg_pixels = arr.reshape(-1, 3)

            b_list.append(float(fg_pixels.mean()))
            rgb_dict["R"].append(float(fg_pixels[:, 0].mean()))
            rgb_dict["G"].append(float(fg_pixels[:, 1].mean()))
            rgb_dict["B"].append(float(fg_pixels[:, 2].mean()))

    axes[0].hist(fresh_brightness, bins=25, alpha=0.65, label="Fresh (Quả tươi)", color="#2ecc71", density=True)
    axes[0].hist(rotten_brightness, bins=25, alpha=0.65, label="Rotten (Quả hỏng)", color="#e74c3c", density=True)
    axes[0].set_title("Phân phối Độ sáng Vùng Tiền cảnh Quả (Foreground Intensity)")
    axes[0].set_xlabel("Cường độ sáng trung bình vùng quả [0, 1]")
    axes[0].set_ylabel("Mật độ (Density)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    x_pos = np.arange(3)
    fresh_means = [np.mean(fresh_rgb_means[c]) for c in ("R", "G", "B")]
    rotten_means = [np.mean(rotten_rgb_means[c]) for c in ("R", "G", "B")]
    width = 0.35
    axes[1].bar(x_pos - width / 2, fresh_means, width, label="Fresh (Quả tươi)", color="#2ecc71", edgecolor="black")
    axes[1].bar(x_pos + width / 2, rotten_means, width, label="Rotten (Quả hỏng)", color="#e74c3c", edgecolor="black")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(["Kênh Đỏ (Red)", "Kênh Xanh lá (Green)", "Kênh Xanh dương (Blue)"])
    axes[1].set_ylabel("Giá trị trung bình kênh màu vùng quả [0, 1]")
    axes[1].set_title("So sánh Cường độ 3 kênh màu R-G-B Vùng Quả (Fresh vs. Rotten)")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    f5_path = out_path / "05_fresh_vs_rotten_color_profile.png"
    fig.savefig(f5_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(f5_path)

    # ------------------------------------------------------------------
    # Hình 6: Trực quan hóa Pipeline Data Augmentation trên 1 ảnh mẫu gốc
    # ------------------------------------------------------------------
    sample_row = df.iloc[0]
    raw_orig_img = load_rgb_image_safe(sample_row["filepath"])
    display_orig_img = raw_orig_img.resize((img_size, img_size))
    train_tfm = get_transforms(img_size=img_size)["train"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    axes[0].imshow(display_orig_img)
    axes[0].set_title(f"Ảnh Gốc (Original)\n[{sample_row['label']}]", color="navy", fontweight="bold")
    axes[0].axis("off")

    mean_arr = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std_arr = np.array(IMAGENET_STD).reshape(1, 1, 3)

    torch.manual_seed(42)
    for idx in range(1, 8):
        aug_tensor = train_tfm(raw_orig_img)
        aug_np = aug_tensor.permute(1, 2, 0).numpy() * std_arr + mean_arr
        aug_np = np.clip(aug_np, 0.0, 1.0)
        axes[idx].imshow(aug_np)
        axes[idx].set_title(f"Biến thể Augmentation #{idx}", fontsize=10)
        axes[idx].axis("off")

    plt.suptitle(
        "Minh họa Pipeline Tăng cường dữ liệu (Rotate nền trắng, Crop, Flip, ColorJitter nhẹ, RandomErasing)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    f6_path = out_path / "06_augmentation_pipeline_preview.png"
    fig.savefig(f6_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(f6_path)

    print(f"[EDA COMPLETED] Đã xuất 6 biểu đồ 300 DPI tại thư mục: {out_path.resolve()}")
    return saved_files
