(() => {
  const $ = (id) => document.getElementById(id);
  const image = $("source-image");
  const canvas = $("box-canvas");
  const ctx = canvas.getContext("2d");
  const state = { images: [], index: 0, view: "", boxes: [], selected: -1, dirty: false, pointer: null };

  function current() { return state.images[state.index]; }
  function setStatus(message) { $("save-status").textContent = message; }
  function setDirty() {
    state.dirty = true;
    setStatus("Unsaved changes.");
  }
  function boxAt(x, y) {
    for (let i = state.boxes.length - 1; i >= 0; i--) {
      const b = state.boxes[i];
      if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) return i;
    }
    return -1;
  }
  function draw() {
    if (!image.naturalWidth) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    state.boxes.forEach((b, i) => {
      const x = b[0] * canvas.width, y = b[1] * canvas.height;
      const w = (b[2] - b[0]) * canvas.width, h = (b[3] - b[1]) * canvas.height;
      const color = i === state.selected ? "#ffe14a" : "#36e0ff";
      ctx.fillStyle = i === state.selected ? "rgba(255,225,74,.20)" : "rgba(54,224,255,.16)";
      ctx.fillRect(x, y, w, h);
      // Dark halo plus bright stroke keeps the box visible on both light and dark cutters.
      ctx.lineWidth = Math.max(5, canvas.width / 350);
      ctx.strokeStyle = "rgba(0,0,0,.92)";
      ctx.strokeRect(x, y, w, h);
      ctx.lineWidth = Math.max(2.5, canvas.width / 650);
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = color;
      ctx.font = `bold ${Math.max(14, canvas.width / 55)}px system-ui`;
      ctx.fillText(String(i + 1), b[0] * canvas.width + 5, b[1] * canvas.height + 20);
    });
    $("box-count").textContent = state.boxes.length;
    $("delete-box").disabled = state.selected < 0;
  }
  function pointerPosition(event) {
    // Normalize pointer coordinates against the displayed image, not the outer stage.
    const rect = image.getBoundingClientRect();
    return [
      Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))
    ];
  }
  function renderCurrent() {
    const item = current();
    if (!item) return;
    state.view = item.view || "";
    state.boxes = (item.target_current ? (item.boxes || []) : []).map(b => b.map(Number));
    state.selected = -1;
    state.dirty = false;
    image.hidden = true;
    canvas.hidden = true;
    $("empty-state").hidden = false;
    $("empty-state").textContent = "Loading image…";
    image.src = `/api/top-isometric/image?key=${encodeURIComponent(item.key)}`;
    $("position").textContent = `${state.index + 1} / ${state.images.length} · ${state.images.filter(x => x.annotated).length} annotated`;
    $("image-name").textContent = `${item.name} · ${item.width} × ${item.height}`;
    $("view-status").textContent = state.view ? `Assigned: ${state.view.toUpperCase()}` : "View not assigned";
    $("view-top").classList.toggle("view-active", state.view === "top");
    $("view-isometric").classList.toggle("view-active", state.view === "isometric");
    setStatus(item.annotated ? "Saved current-target annotation loaded." : (item.target_current === false ? "Legacy diamond-face boxes are not valid for these targets. Redraw this image." : "Not yet annotated."));
    draw();
  }
  image.addEventListener("load", () => {
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    image.hidden = false;
    canvas.hidden = false;
    $("empty-state").hidden = true;
    draw();
  });
  image.addEventListener("error", () => {
    image.hidden = true;
    canvas.hidden = true;
    $("empty-state").hidden = false;
    $("empty-state").textContent = "Could not load this image. Try moving to another image and back.";
  });

  canvas.addEventListener("pointerdown", event => {
    if (!image.naturalWidth) return;
    const [x, y] = pointerPosition(event);
    const hit = boxAt(x, y);
    state.selected = hit;
    state.pointer = { startX: x, startY: y, original: hit >= 0 ? state.boxes[hit].slice() : null, moved: false };
    canvas.setPointerCapture(event.pointerId);
    draw();
  });
  canvas.addEventListener("pointermove", event => {
    if (!state.pointer) return;
    const [x, y] = pointerPosition(event);
    const p = state.pointer;
    if (p.original) {
      const dx = x - p.startX, dy = y - p.startY;
      if (Math.abs(dx) + Math.abs(dy) > 0.001) p.moved = true;
      const b = p.original;
      const w = b[2] - b[0], h = b[3] - b[1];
      const x1 = Math.max(0, Math.min(1 - w, b[0] + dx));
      const y1 = Math.max(0, Math.min(1 - h, b[1] + dy));
      state.boxes[state.selected] = [x1, y1, x1 + w, y1 + h];
    } else {
      p.endX = x; p.endY = y; p.moved = true;
    }
    draw();
    if (!p.original && p.endX !== undefined) {
      const rect = [Math.min(p.startX, p.endX), Math.min(p.startY, p.endY),
        Math.max(p.startX, p.endX), Math.max(p.startY, p.endY)];
      ctx.strokeStyle = "#36d7ff";
      ctx.lineWidth = Math.max(2, canvas.width / 700);
      ctx.setLineDash([8, 6]);
      ctx.strokeRect(rect[0] * canvas.width, rect[1] * canvas.height,
        (rect[2] - rect[0]) * canvas.width, (rect[3] - rect[1]) * canvas.height);
      ctx.setLineDash([]);
    }
  });
  function finishPointer() {
    if (!state.pointer) return;
    const p = state.pointer;
    if (!p.original && p.endX !== undefined) {
      const box = [Math.min(p.startX, p.endX), Math.min(p.startY, p.endY),
        Math.max(p.startX, p.endX), Math.max(p.startY, p.endY)];
      if (box[2] - box[0] >= 0.003 && box[3] - box[1] >= 0.003) {
        state.boxes.push(box);
        state.selected = state.boxes.length - 1;
        setDirty();
      }
    } else if (p.original && p.moved) {
      setDirty();
    }
    state.pointer = null;
    draw();
  }
  canvas.addEventListener("pointerup", finishPointer);
  canvas.addEventListener("pointercancel", finishPointer);

  async function saveCurrent() {
    const item = current();
    if (!item || !state.dirty) return true;
    if (state.boxes.length && !["top", "isometric"].includes(state.view)) {
      setStatus("Choose Top or Isometric before saving boxes.");
      return false;
    }
    const response = await fetch("/api/top-isometric/save", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: item.key, view: state.view, boxes: state.boxes })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.description || data.error || "Save failed.");
    item.view = state.boxes.length ? state.view : "";
    item.boxes = state.boxes.map(b => b.slice());
    item.annotated = state.boxes.length > 0;
    item.target_current = true;
    state.dirty = false;
    $("position").textContent = `${state.index + 1} / ${state.images.length} · ${state.images.filter(x => x.annotated).length} annotated`;
    setStatus("Saved to dataset/top_isometric_annotations.json.");
    return true;
  }
  async function move(delta) {
    try {
      if (!(await saveCurrent())) return;
      const next = state.index + delta;
      if (next < 0 || next >= state.images.length) return;
      state.index = next;
      renderCurrent();
    } catch (error) { setStatus(error.message); }
  }
  $("previous").addEventListener("click", () => move(-1));
  $("next").addEventListener("click", () => move(1));
  $("view-top").addEventListener("click", () => {
    state.view = "top"; $("view-status").textContent = "Assigned: TOP";
    $("view-top").classList.add("view-active"); $("view-isometric").classList.remove("view-active"); setDirty();
  });
  $("view-isometric").addEventListener("click", () => {
    state.view = "isometric"; $("view-status").textContent = "Assigned: ISOMETRIC";
    $("view-isometric").classList.add("view-active"); $("view-top").classList.remove("view-active"); setDirty();
  });
  $("delete-box").addEventListener("click", () => {
    if (state.selected < 0) return;
    state.boxes.splice(state.selected, 1); state.selected = -1; setDirty(); draw();
  });
  $("clear-boxes").addEventListener("click", () => {
    if (!state.boxes.length) return;
    state.boxes = []; state.selected = -1; setDirty(); draw();
  });
  $("save").addEventListener("click", async () => {
    try { await saveCurrent(); } catch (error) { setStatus(error.message); }
  });
  $("export").addEventListener("click", async () => {
    try {
      if (!(await saveCurrent())) return;
      $("export-status").textContent = "Exporting…";
      const response = await fetch("/api/top-isometric/export", { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.description || "Export failed.");
      $("export-status").textContent = `Export complete: ${data.annotation_count} saved annotation record(s) processed into training/.`;
    } catch (error) { $("export-status").textContent = error.message; }
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Delete" && state.selected >= 0) $("delete-box").click();
    if (event.key === "ArrowLeft") move(-1);
    if (event.key === "ArrowRight") move(1);
  });

  fetch("/api/top-isometric/images")
    .then(response => response.json())
    .then(data => {
      state.images = data.images || [];
      if (!state.images.length) {
        $("empty-state").hidden = false;
        $("empty-state").textContent = data.error || "No images found in dataset/train/images.";
        $("position").textContent = "0 images";
        return;
      }
      renderCurrent();
    })
    .catch(error => {
      $("empty-state").hidden = false;
      $("empty-state").textContent = `Could not load image list: ${error.message}`;
      $("position").textContent = "Image list unavailable";
    });
})();
