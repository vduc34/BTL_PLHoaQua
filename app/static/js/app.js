/**
 * Giai đoạn 4 (v2.0 - Đã hiệu đính theo Kiểm định Chéo):
 * Frontend Controller cho FruitAI Quality & Multi-Class Inspector.
 * Hỗ trợ: Drag & Drop, Webcam trực tiếp, Thanh trượt Opacity Grad-CAM,
 * Tự động dự đoán lại khi đổi mô hình (#archSelect), Phân biệt minh bạch Trọng số Colab vs Demo Local,
 * Đấu trường so sánh 3 mô hình (Ablation Live), Kiểm định lô hàng (Batch) & Xuất CSV.
 */

let currentSelectedFile = null;
let currentSampleFilename = "";
let webcamStream = null;
let inspectionHistory = [];

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initDropzone();
  initWebcamControls();
  initGradcamSlider();
  initBatchAndHistory();

  fetchModelsStatus();
  fetchSampleImages();
  fetchReportFigures();

  document.getElementById("btnReloadModels").addEventListener("click", async () => {
    const btn = document.getElementById("btnReloadModels");
    btn.textContent = "⏳ Đang quét models/...";
    try {
      const res = await fetch("/api/reload-models", { method: "POST" });
      const data = await res.json();
      renderModelStatusStrip(data.status);
      if (currentSelectedFile || currentSampleFilename) {
        runSinglePrediction();
      }
    } finally {
      btn.textContent = "🔄 Nạp lại Trọng số (.pth)";
    }
  });

  // Tự động chạy lại suy luận ngay khi người dùng đổi mô hình trong Dropdown
  document.getElementById("archSelect").addEventListener("change", () => {
    if (currentSelectedFile || currentSampleFilename) {
      runSinglePrediction();
    }
  });

  document.getElementById("btnPredictSingle").addEventListener("click", () => {
    runSinglePrediction();
  });

  document.getElementById("btnRunCompare").addEventListener("click", () => {
    runThreeModelsComparison();
  });

  document.getElementById("btnRefreshFigures").addEventListener("click", () => {
    fetchReportFigures();
  });
});

/* ========================================================================== */
/* 1. QUẢN LÝ CHUYỂN TAB                                                      */
/* ========================================================================== */
function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  const panes = document.querySelectorAll(".tab-pane");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      panes.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add("active");
    });
  });
}

/* ========================================================================== */
/* 2. TRẠNG THÁI 3 MÔ HÌNH & BỘ ẢNH MẪU                                       */
/* ========================================================================== */
async function fetchModelsStatus() {
  const res = await fetch("/api/models");
  const data = await res.json();
  renderModelStatusStrip(data);
}

function renderModelStatusStrip(statusData) {
  document.getElementById("deviceText").textContent = `Thiết bị: ${statusData.device.toUpperCase()}`;
  const strip = document.getElementById("modelStatusStrip");
  strip.innerHTML = "";

  const displayNames = {
    efficientnet_b0: "EfficientNet-B0 (SE-Attention)",
    mobilenet_v2: "MobileNetV2 (Lightweight)",
    custom_cnn: "Custom CNN (Baseline 4-Conv)",
  };

  Object.entries(statusData.models).forEach(([arch, meta]) => {
    const isColabTrained = meta.loaded_from_checkpoint && !meta.is_demo_checkpoint;
    const badgeClass = isColabTrained ? "chip-ready" : "chip-demo";
    const badgeText = isColabTrained
      ? `✓ Trọng số Colab (.pth) | F1: ${meta.best_val_f1}%`
      : "⚡ Bản Demo Local (Đã hiệu chuẩn mẫu)";

    const chip = document.createElement("div");
    chip.className = "model-chip";
    chip.innerHTML = `
      <div>
        <h4>${displayNames[arch] || arch.toUpperCase()}</h4>
        <small>${meta.total_params.toLocaleString()} params (${meta.file_size_mb} MB) | ${meta.checkpoint_file}</small>
      </div>
      <span class="chip-tag ${badgeClass}">
        ${badgeText}
      </span>
    `;
    strip.appendChild(chip);
  });
}

async function fetchSampleImages() {
  const res = await fetch("/api/sample-images");
  const data = await res.json();
  const grid = document.getElementById("samplesGrid");
  grid.innerHTML = "";

  (data.samples || []).forEach((item, idx) => {
    const div = document.createElement("div");
    div.className = "sample-item";
    div.innerHTML = `
      <img src="${item.url}" alt="${item.filename}" />
      <span>${item.label_hint}</span>
    `;
    div.addEventListener("click", () => {
      document.querySelectorAll(".sample-item").forEach((el) => el.classList.remove("active"));
      div.classList.add("active");
      currentSelectedFile = null;
      currentSampleFilename = item.filename;

      const preview = document.getElementById("previewInputImg");
      preview.src = item.url;
      preview.classList.remove("hidden");
      document.getElementById("dropzonePrompt").classList.add("hidden");
      runSinglePrediction();
    });
    grid.appendChild(div);

    if (idx === 0 && !currentSelectedFile && !currentSampleFilename) {
      div.click();
    }
  });
}

/* ========================================================================== */
/* 3. KÉO THẢ ẢNH (DRAG & DROP) & CHỤP CAMERA WEBCAM                          */
/* ========================================================================== */
function initDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleUserSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleUserSelectedFile(e.target.files[0]);
    }
  });
}

function handleUserSelectedFile(file) {
  currentSelectedFile = file;
  currentSampleFilename = "";
  document.querySelectorAll(".sample-item").forEach((el) => el.classList.remove("active"));

  const reader = new FileReader();
  reader.onload = (ev) => {
    const preview = document.getElementById("previewInputImg");
    preview.src = ev.target.result;
    preview.classList.remove("hidden");
    document.getElementById("dropzonePrompt").classList.add("hidden");
  };
  reader.readAsDataURL(file);
  runSinglePrediction();
}

function initWebcamControls() {
  const btnToggle = document.getElementById("btnToggleWebcam");
  const btnCapture = document.getElementById("btnCaptureWebcam");
  const btnClose = document.getElementById("btnCloseWebcam");
  const webcamBox = document.getElementById("webcamBox");
  const video = document.getElementById("webcamVideo");
  const canvas = document.getElementById("webcamCanvas");

  btnToggle.addEventListener("click", async () => {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
      video.srcObject = webcamStream;
      webcamBox.classList.remove("hidden");
    } catch (err) {
      alert("Không thể truy cập Camera: " + err.message);
    }
  });

  btnClose.addEventListener("click", () => {
    stopWebcam();
  });

  btnCapture.addEventListener("click", () => {
    if (!webcamStream) return;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (blob) {
        const capturedFile = new File([blob], `webcam_${Date.now()}.png`, { type: "image/png" });
        stopWebcam();
        handleUserSelectedFile(capturedFile);
      }
    }, "image/png");
  });
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
  document.getElementById("webcamBox").classList.add("hidden");
}

function initGradcamSlider() {
  const slider = document.getElementById("gradcamOpacitySlider");
  const valText = document.getElementById("opacityVal");
  const heatmapImg = document.getElementById("resPureHeatmapImg");

  slider.addEventListener("input", (e) => {
    const pct = parseInt(e.target.value, 10);
    valText.textContent = `${pct}%`;
    heatmapImg.style.opacity = (pct / 100.0).toFixed(2);
  });
}

/* ========================================================================== */
/* 4. SUY LUẬN ĐƠN ẢNH & SO SÁNH 3 MÔ HÌNH                                    */
/* ========================================================================== */
async function runSinglePrediction() {
  if (!currentSelectedFile && !currentSampleFilename) {
    alert("Vui lòng chọn 1 ảnh hoa quả trước!");
    return;
  }

  const arch = document.getElementById("archSelect").value;
  const formData = new FormData();
  formData.append("arch_name", arch);
  if (currentSelectedFile) {
    formData.append("file", currentSelectedFile);
  } else {
    formData.append("sample_filename", currentSampleFilename);
  }

  const btn = document.getElementById("btnPredictSingle");
  btn.textContent = "⏳ Đang phân tích & tính Grad-CAM...";
  btn.disabled = true;

  try {
    const res = await fetch("/api/predict", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Lỗi suy luận");
    }
    const data = await res.json();
    renderSinglePredictionResult(data);
    appendHistoryItem(data);
  } catch (err) {
    alert("Lỗi: " + err.message);
  } finally {
    btn.textContent = "🚀 Phân Loại & Trích Xuất Grad-CAM";
    btn.disabled = false;
  }
}

function renderSinglePredictionResult(data) {
  document.getElementById("emptyResultState").classList.add("hidden");
  document.getElementById("filledResultState").classList.remove("hidden");

  document.getElementById("resLatency").textContent = `⏱ ${data.latency_ms} ms (${data.arch_name.toUpperCase()})`;

  const banner = document.getElementById("verdictBanner");
  banner.className = `verdict-banner ${data.status === "fresh" ? "verdict-fresh" : "verdict-rotten"}`;
  document.getElementById("resStatusBadge").textContent = data.status_badge;
  document.getElementById("resViName").textContent = data.vi_name;
  const modeNote = data.is_demo_checkpoint ? "Bản Demo Local" : "Trọng số Colab (.pth)";
  document.getElementById("resModelInfo").textContent = `Mô hình: ${data.arch_name.toUpperCase()} (${modeNote}) | Nhãn chuẩn: ${data.predicted_class}`;
  document.getElementById("resConfidence").textContent = `${data.confidence_percent.toFixed(1)}%`;

  document.getElementById("resFreshScore").textContent = `${data.freshness_score_percent.toFixed(1)}%`;
  document.getElementById("resSpoilScore").textContent = `${data.spoilage_risk_percent.toFixed(1)}%`;
  document.getElementById("barFresh").style.width = `${data.freshness_score_percent}%`;
  document.getElementById("barRotten").style.width = `${data.spoilage_risk_percent}%`;

  document.getElementById("resOrigImg").src = data.images.original_b64;
  document.getElementById("resBaseImgForStack").src = data.images.original_b64;
  document.getElementById("resPureHeatmapImg").src = data.images.gradcam_pure_b64;

  const top3Container = document.getElementById("top3Container");
  top3Container.innerHTML = "";
  (data.top3 || []).forEach((item) => {
    const color = item.status === "fresh" ? "#10b981" : "#ef4444";
    const row = document.createElement("div");
    row.className = "top3-row";
    row.innerHTML = `
      <div class="top3-meta">
        <span><strong>${item.vi_name}</strong> <code>(${item.class_label})</code></span>
        <strong>${item.confidence_percent.toFixed(2)}%</strong>
      </div>
      <div class="top3-track">
        <div class="top3-fill" style="width: ${item.confidence_percent}%; background: ${color};"></div>
      </div>
    `;
    top3Container.appendChild(row);
  });

  const kn = data.domain_knowledge || {};
  document.getElementById("knCalories").textContent = kn.calories_per_100g || "--";
  document.getElementById("knStorage").textContent = kn.storage_temp || "--";
  document.getElementById("knShelfLife").textContent = kn.shelf_life || "--";
  document.getElementById("knVitamins").textContent = kn.vitamins || "--";
  document.getElementById("knAssessment").textContent = kn.quality_assessment || "--";
  document.getElementById("knRecommendation").textContent = kn.recommendation || "--";
}

async function runThreeModelsComparison() {
  if (!currentSelectedFile && !currentSampleFilename) {
    alert("Vui lòng chọn 1 ảnh ở Tab 1 trước khi chạy so sánh 3 mô hình!");
    return;
  }

  const formData = new FormData();
  if (currentSelectedFile) {
    formData.append("file", currentSelectedFile);
  } else {
    formData.append("sample_filename", currentSampleFilename);
  }

  const btn = document.getElementById("btnRunCompare");
  btn.textContent = "⏳ Đang chạy qua cả 3 mạng CNN...";
  btn.disabled = true;

  try {
    const res = await fetch("/api/compare-models", { method: "POST", body: formData });
    const data = await res.json();

    const banner = document.getElementById("compareSummaryBanner");
    banner.classList.remove("hidden");
    banner.textContent = `🤝 Mức độ đồng thuận giữa 3 kiến trúc (Consensus): ${data.consensus_ratio} mô hình cùng dự đoán một nhãn trên file [${data.filename}].`;

    const grid = document.getElementById("compareArenaGrid");
    grid.innerHTML = "";

    (data.results || []).forEach((r) => {
      const isFresh = r.status === "fresh";
      const card = document.createElement("div");
      card.className = "arena-card";
      card.innerHTML = `
        <h3>🧠 ${r.arch_name.toUpperCase()}</h3>
        <span class="badge-status" style="background: ${isFresh ? "#10b981" : "#ef4444"}; color: #fff;">
          ${r.status_badge}
        </span>
        <p style="margin-top: 0.4rem;"><strong>Dự đoán:</strong> ${r.vi_name}</p>
        <p><strong>Độ tự tin:</strong> ${r.confidence_percent.toFixed(2)}% | <strong>Tốc độ:</strong> ${r.latency_ms} ms</p>
        <img class="arena-img" src="${r.images.gradcam_overlay_b64}" alt="Grad-CAM ${r.arch_name}" />
        <small style="color:#94a3b8;">Bản đồ nhiệt Grad-CAM trên tầng <code>features[-1]</code></small>
      `;
      grid.appendChild(card);
    });
  } catch (err) {
    alert("Lỗi so sánh: " + err.message);
  } finally {
    btn.textContent = "⚡ Chạy So Sánh Cả 3 Mô Hình (Trên Ảnh Đang Chọn)";
    btn.disabled = false;
  }
}

/* ========================================================================== */
/* 5. KIỂM ĐỊNH LÔ HÀNG (BATCH) & LỊCH SỬ XUẤT CSV                            */
/* ========================================================================== */
function initBatchAndHistory() {
  const batchInput = document.getElementById("batchFileInput");
  const btnBatch = document.getElementById("btnSelectBatchFiles");
  btnBatch.addEventListener("click", () => batchInput.click());

  batchInput.addEventListener("change", async (e) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const arch = document.getElementById("archSelect").value;
    const formData = new FormData();
    formData.append("arch_name", arch);
    Array.from(e.target.files).forEach((f) => formData.append("files", f));

    btnBatch.textContent = `⏳ Đang kiểm định ${e.target.files.length} ảnh...`;
    btnBatch.disabled = true;
    try {
      const res = await fetch("/api/batch-predict", { method: "POST", body: formData });
      const data = await res.json();
      (data.items || []).forEach((item) => {
        if (!item.error) appendHistoryItem(item);
      });
    } catch (err) {
      alert("Lỗi kiểm định lô hàng: " + err.message);
    } finally {
      btnBatch.textContent = "📂 Chọn Nhiều Ảnh Lô Hàng";
      btnBatch.disabled = false;
      batchInput.value = "";
    }
  });

  document.getElementById("btnClearHistory").addEventListener("click", () => {
    inspectionHistory = [];
    renderHistoryTable();
  });

  document.getElementById("btnExportHistoryCsv").addEventListener("click", () => {
    if (inspectionHistory.length === 0) {
      alert("Lịch sử kiểm định đang trống!");
      return;
    }
    const headers = ["STT", "File", "Model", "Predicted_Class", "Vietnamese_Name", "Status", "Confidence_Pct", "Latency_ms", "Timestamp"];
    const rows = inspectionHistory.map((h, i) => [
      i + 1,
      `"${h.filename}"`,
      h.arch_name,
      h.predicted_class,
      `"${h.vi_name}"`,
      h.status.toUpperCase(),
      h.confidence_percent,
      h.latency_ms,
      `"${h.timestamp}"`,
    ]);
    const csvContent = "\uFEFF" + [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `Bao_Cao_Kiem_Dinh_Hoa_Qua_${Date.now()}.csv`;
    link.click();
  });
}

function appendHistoryItem(pred) {
  inspectionHistory.unshift({
    filename: pred.filename || "image.png",
    arch_name: pred.arch_name,
    predicted_class: pred.predicted_class,
    vi_name: pred.vi_name,
    status: pred.status,
    status_badge: pred.status_badge,
    confidence_percent: pred.confidence_percent,
    latency_ms: pred.latency_ms,
    thumb: pred.images.original_b64,
    timestamp: new Date().toLocaleTimeString("vi-VN"),
  });
  renderHistoryTable();
}

function renderHistoryTable() {
  const total = inspectionHistory.length;
  const freshCnt = inspectionHistory.filter((x) => x.status === "fresh").length;
  const rottenCnt = total - freshCnt;
  const passRate = total > 0 ? ((freshCnt / total) * 100.0).toFixed(1) : "0.0";

  document.getElementById("kpiTotal").textContent = total;
  document.getElementById("kpiFresh").textContent = freshCnt;
  document.getElementById("kpiRotten").textContent = rottenCnt;
  document.getElementById("kpiPassRate").textContent = `${passRate}%`;

  const tbody = document.getElementById("historyTableBody");
  if (total === 0) {
    tbody.innerHTML = `<tr><td colspan="9" class="text-center">Chưa có mẫu nào được kiểm định trong phiên này.</td></tr>`;
    return;
  }

  tbody.innerHTML = inspectionHistory
    .map(
      (item, idx) => `
    <tr>
      <td>${idx + 1}</td>
      <td><img class="thumb-sm" src="${item.thumb}" alt="thumb" /></td>
      <td>${item.filename}</td>
      <td><code>${item.arch_name.toUpperCase()}</code></td>
      <td><strong>${item.vi_name}</strong></td>
      <td>
        <span class="chip-tag ${item.status === "fresh" ? "chip-ready" : "chip-demo"}" style="color:${item.status === "fresh" ? "#10b981" : "#ef4444"}">
          ${item.status === "fresh" ? "TƯƠI (SAFE)" : "HỎNG (REJECT)"}
        </span>
      </td>
      <td>${item.confidence_percent.toFixed(1)}%</td>
      <td>${item.latency_ms} ms</td>
      <td>${item.timestamp}</td>
    </tr>
  `
    )
    .join("");
}

/* ========================================================================== */
/* 6. THƯ VIỆN 12 BIỂU ĐỒ BÁO CÁO (300 DPI)                                   */
/* ========================================================================== */
async function fetchReportFigures() {
  const res = await fetch("/api/report-figures");
  const data = await res.json();
  const gallery = document.getElementById("reportFiguresGallery");
  gallery.innerHTML = "";

  const figs = data.figures || [];
  if (figs.length === 0) {
    gallery.innerHTML = `
      <div class="empty-state" style="grid-column: 1 / -1;">
        <p>Chưa có biểu đồ trong <code>reports/figures/</code>. Hãy giải nén gói kết quả từ Google Colab vào thư mục dự án!</p>
      </div>
    `;
    return;
  }

  figs.forEach((f) => {
    const card = document.createElement("div");
    card.className = "figure-card";
    card.innerHTML = `
      <h4>🖼️ ${f.filename}</h4>
      <img src="${f.url}?t=${Date.now()}" alt="${f.filename}" loading="lazy" />
    `;
    gallery.appendChild(card);
  });
}
