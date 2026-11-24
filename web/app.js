// Demo Pix2Pix · Juan Sebastian Castillo Acevedo (@Juansec01)
// Envía la imagen a POST /api/restore y muestra el PNG que devuelve el servidor.
"use strict";

const $ = (id) => document.getElementById(id);
const fileInput = $("file");
const drop = $("drop");
const restoreBtn = $("restore");
const before = $("before");
const after = $("after");
const spinner = $("spinner");
const download = $("download");
const message = $("message");
const statusBadge = $("status");

let currentFile = null;
let currentExample = null; // nombre del ejemplo de test elegido (tiene original)
const exampleIndex = new Map(); // nombre de archivo -> ejemplo (para fotos arrastradas)
let afterUrl = null;
const original = $("original");
const figOriginal = $("fig-original");
const compare = $("compare");
const current = $("current");
const mBefore = $("m-before");
const mAfter = $("m-after");
const delta = $("delta");

function fmt(m) {
  return `PSNR ${m.psnr.toFixed(2)} dB · SSIM ${m.ssim.toFixed(3)}`;
}

function clearMetrics() {
  mBefore.textContent = "";
  mAfter.textContent = "";
  delta.hidden = true;
}

function showOriginal(url) {
  if (url) {
    original.src = url;
    figOriginal.hidden = false;
    compare.classList.add("con-original");
  } else {
    original.removeAttribute("src");
    figOriginal.hidden = true;
    compare.classList.remove("con-original");
  }
}

function showMetrics(header) {
  if (!header) return;
  const m = JSON.parse(header);
  mBefore.textContent = fmt(m.entrada);
  mAfter.textContent = fmt(m.restaurada);
  const d = m.restaurada.psnr - m.entrada.psnr;
  const ds = m.restaurada.ssim - m.entrada.ssim;
  const db = `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(2)} dB`;
  const ss = `${ds >= 0 ? "+" : "−"}${Math.abs(ds).toFixed(3)}`;
  if (d >= 0) {
    delta.textContent = `La restaurada se parece más a la original: ${db} de PSNR, ${ss} de SSIM.`;
  } else if (ds > 0) {
    delta.textContent =
      `El PSNR baja (${db}) pero la estructura mejora (SSIM ${ss}): la entrada estaba poco ` +
      "dañada y el modelo cambia el color y el brillo más de lo necesario.";
  } else {
    delta.textContent =
      `En esta foto el modelo no mejora (${db} de PSNR, SSIM ${ss}): la entrada estaba poco ` +
      "dañada o el daño es desenfoque.";
  }
  delta.classList.toggle("peor", d < 0);
  delta.hidden = false;
}

function say(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("error", isError);
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.model_loaded || data.checkpoint) {
      statusBadge.textContent = data.model_loaded ? "Modelo listo" : "Modelo se carga al restaurar";
      statusBadge.className = "badge ok";
    } else {
      statusBadge.textContent = "Sin checkpoint configurado";
      statusBadge.className = "badge warn";
    }
  } catch {
    statusBadge.textContent = "Servidor no disponible";
    statusBadge.className = "badge warn";
  }
}

async function loadExamples() {
  try {
    const res = await fetch("/api/examples");
    const items = await res.json();
    if (!items.length) return;
    const box = $("examples");
    for (const item of items) {
      exampleIndex.set(item.name, item);
      const button = document.createElement("button");
      button.type = "button";
      button.title = item.name;
      button.setAttribute("role", "listitem");
      button.setAttribute("aria-pressed", "false");
      const img = document.createElement("img");
      img.src = item.url;
      img.alt = item.name;
      img.loading = "lazy";
      button.append(img);
      button.addEventListener("click", async () => {
        box.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));
        button.setAttribute("aria-pressed", "true");
        const blob = await (await fetch(item.url)).blob();
        setFile(new File([blob], item.name, { type: blob.type || "image/png" }), item);
      });
      box.append(button);
    }
    $("examples-box").hidden = false;
  } catch {
    /* sin ejemplos: la demo sigue funcionando con imágenes propias */
  }
}

function setFile(file, example = null) {
  if (!file) return;
  // Una foto del test arrastrada desde el explorador también se compara con su original.
  if (!example && exampleIndex.has(file.name)) example = exampleIndex.get(file.name);
  if (!file.type.startsWith("image/")) {
    say("El archivo no es una imagen.", true);
    return;
  }
  currentFile = file;
  currentExample = example && example.original_url ? example.name : null;
  showOriginal(example ? example.original_url : null);
  clearMetrics();
  current.innerHTML = "";
  const code = document.createElement("code");
  code.textContent = file.name;
  current.append("Foto: ", code);
  current.hidden = false;
  if (!example || !exampleIndex.has(file.name)) {
    $("examples")
      .querySelectorAll("button")
      .forEach((b) => b.setAttribute("aria-pressed", "false"));
  }
  before.src = URL.createObjectURL(file);
  after.removeAttribute("src");
  download.hidden = true;
  restoreBtn.disabled = false;
  say(`${file.name} · ${(file.size / 1024).toFixed(0)} KB`);
}

async function restore() {
  if (!currentFile) return;
  restoreBtn.disabled = true;
  spinner.hidden = false;
  say("Restaurando…");
  const body = new FormData();
  body.append("file", currentFile);
  if (currentExample) body.append("example", currentExample);
  body.append("limpiar", $("limpiar").checked ? "true" : "false");
  body.append("nitidez", $("nitidez").checked ? "true" : "false");
  try {
    const res = await fetch("/api/restore", { method: "POST", body });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Error ${res.status}`);
    }
    const blob = await res.blob();
    showMetrics(res.headers.get("X-Metricas"));
    if (afterUrl) URL.revokeObjectURL(afterUrl);
    afterUrl = URL.createObjectURL(blob);
    after.src = afterUrl;
    download.href = afterUrl;
    download.hidden = false;
    say("Listo.");
    checkHealth();
loadExamples();
  } catch (error) {
    say(error.message, true);
  } finally {
    spinner.hidden = true;
    restoreBtn.disabled = false;
  }
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
restoreBtn.addEventListener("click", restore);
["dragenter", "dragover"].forEach((type) =>
  drop.addEventListener(type, (event) => {
    event.preventDefault();
    drop.classList.add("over");
  }),
);
["dragleave", "drop"].forEach((type) =>
  drop.addEventListener(type, (event) => {
    event.preventDefault();
    drop.classList.remove("over");
  }),
);
drop.addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));

checkHealth();
loadExamples();
