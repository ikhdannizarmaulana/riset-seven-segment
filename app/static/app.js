// Frontend MVP: tanpa framework agar mudah dipahami & dijelaskan saat sidang.
const $ = (id) => document.getElementById(id);
let file = null, img = null, last = null, drag = null;
const PARAMS = ["roi_x","roi_y","roi_w","roi_h","target_height","grayscale","clahe","clahe_clip",
                "blur","threshold","threshold_value","invert","dilate"];
const META = ["device","lighting","session"];

async function loadEngines() {
  const r = await fetch("/api/engines"); const d = await r.json();
  $("engine").innerHTML = d.engines.map(e =>
    `<option value="${e.name}" ${e.available ? "" : "disabled"}>${e.label}${e.available ? "" : " — " + e.status}</option>`).join("");
  $("lighting").innerHTML = d.lighting_options.map(o => `<option>${o}</option>`).join("");
  META.forEach(k => { const v = localStorage.getItem("meta_" + k); if (v) $(k).value = v; });
}

function formData(extra = {}) {
  const fd = new FormData();
  fd.append("image", file);
  PARAMS.forEach(k => {
    const el = $(k);
    fd.append(k, el.type === "checkbox" ? (el.checked ? "true" : "false") : el.value);
  });
  Object.entries(extra).forEach(([k, v]) => fd.append(k, v));
  return fd;
}

// ---------- ROI di canvas ----------
const cv = $("canvas"), ctx = cv.getContext("2d");
function draw() {
  if (!img) return;
  cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);
  const [x, y, w, h] = ["roi_x","roi_y","roi_w","roi_h"].map(k => +$(k).value);
  if (w > 0 && h > 0) {
    ctx.lineWidth = Math.max(2, cv.width / 300); ctx.strokeStyle = "#e11";
    ctx.strokeRect(x, y, w, h);
  }
}
function toImg(ev) {
  const r = cv.getBoundingClientRect();
  const p = ev.touches ? ev.touches[0] : ev;
  return [Math.round((p.clientX - r.left) * cv.width / r.width),
          Math.round((p.clientY - r.top) * cv.height / r.height)];
}
function start(ev) { if (!img) return; ev.preventDefault(); drag = toImg(ev); }
function move(ev) {
  if (!drag) return; ev.preventDefault();
  const [x2, y2] = toImg(ev), [x1, y1] = drag;
  $("roi_x").value = Math.max(0, Math.min(x1, x2)); $("roi_y").value = Math.max(0, Math.min(y1, y2));
  $("roi_w").value = Math.abs(x2 - x1); $("roi_h").value = Math.abs(y2 - y1); draw();
}
function end() { if (drag) { drag = null; preview(); } }
cv.addEventListener("mousedown", start); cv.addEventListener("mousemove", move);
window.addEventListener("mouseup", end);
cv.addEventListener("touchstart", start, {passive: false});
cv.addEventListener("touchmove", move, {passive: false}); cv.addEventListener("touchend", end);
["roi_x","roi_y","roi_w","roi_h"].forEach(k => $(k).addEventListener("change", draw));
$("roiReset").onclick = () => { ["roi_x","roi_y","roi_w","roi_h"].forEach(k => $(k).value = 0); draw(); preview(); };

$("file").onchange = (e) => {
  file = e.target.files[0]; if (!file) return;
  img = new Image();
  img.onload = () => { $("roiReset").onclick(); };
  img.src = URL.createObjectURL(file);
  $("result").innerHTML = ""; $("btnSave").disabled = true; last = null;
};

// ---------- API ----------
async function preview() {
  if (!file) return;
  const r = await fetch("/api/preprocess", {method: "POST", body: formData()});
  const d = await r.json();
  if (!r.ok) { $("previewInfo").textContent = "Error: " + d.detail; return; }
  $("preview").src = d.preview;
  $("previewInfo").textContent = `Ukuran keluaran ${d.output_size.join("×")} · ${d.preprocess_ms} ms`;
}
$("btnPreview").onclick = preview;
PARAMS.slice(4).forEach(k => $(k).addEventListener("change", preview));

$("btnPredict").onclick = async () => {
  if (!file) { alert("Pilih/ambil foto dulu"); return; }
  $("result").textContent = "Memproses…";
  const r = await fetch("/api/predict", {method: "POST", body: formData({engine: $("engine").value})});
  const d = await r.json();
  if (!r.ok) { $("result").textContent = "Error: " + d.detail; return; }
  last = d;
  $("preview").src = d.preview;
  $("result").innerHTML = `<div class="big">${d.text || "(kosong)"}</div>
    <div>raw: <code>${d.raw || ""}</code> · confidence: ${d.confidence ?? "-"}</div>
    <div>preprocess ${d.preprocess_ms} ms · OCR ${d.ocr_ms} ms · total ${d.total_ms} ms</div>`;
  $("label").value = d.text; $("label").focus(); $("btnSave").disabled = false;
};

$("btnSave").onclick = async () => {
  if (!last) return;
  META.forEach(k => localStorage.setItem("meta_" + k, $(k).value));
  const fd = new FormData();
  const f = {image_file: last.image_file, crop_file: last.crop_file, label: $("label").value.trim(),
    predicted: last.text, engine: last.engine, confidence: last.confidence ?? "",
    ocr_ms: last.ocr_ms, preprocess_ms: last.preprocess_ms, device: $("device").value,
    lighting: $("lighting").value, session: $("session").value, notes: $("notes").value,
    params_json: JSON.stringify(last.params), roi_x: last.params.roi_x, roi_y: last.params.roi_y,
    roi_w: last.params.roi_w, roi_h: last.params.roi_h};
  Object.entries(f).forEach(([k, v]) => fd.append(k, v));
  const r = await fetch("/api/label", {method: "POST", body: fd});
  const d = await r.json();
  if (!r.ok) { $("saveInfo").textContent = "Error: " + d.detail; return; }
  $("saveInfo").textContent = `Tersimpan (${d.row.correct === 1 ? "prediksi benar" : "dikoreksi"}). Total data: ${d.total}`;
  $("btnSave").disabled = true; $("notes").value = "";
};

loadEngines();
