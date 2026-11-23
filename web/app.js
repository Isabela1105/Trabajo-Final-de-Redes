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
let afterUrl = null;

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
        setFile(new File([blob], item.name, { type: blob.type || "image/png" }));
      });
      box.append(button);
    }
    $("examples-box").hidden = false;
  } catch {
    /* sin ejemplos: la demo sigue funcionando con imágenes propias */
  }
}

function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    say("El archivo no es una imagen.", true);
    return;
  }
  currentFile = file;
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
  try {
    const res = await fetch("/api/restore", { method: "POST", body });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Error ${res.status}`);
    }
    const blob = await res.blob();
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
