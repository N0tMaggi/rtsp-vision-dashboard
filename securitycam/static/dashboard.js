(function () {
  const VISUAL_TOGGLES = [
    ["boxes", "Boxes"],
    ["names", "Names"],
    ["conf", "Confidence"],
    ["skeleton", "Skeleton"],
    ["head", "Head"],
    ["snaplines", "Snaplines"],
    ["center_dot", "Crosshair"],
    ["distance", "Distance"],
    ["face_boxes", "Face Boxes"],
    ["face_zoom", "Face Zoom"],
    ["hand_skeleton", "Hand Skeleton"],
    ["chams", "Chams"],
    ["face_blur", "Face Blur"],
    ["fill_box", "Box Fill"],
    ["class_colors", "Class Colors"],
    ["person_only", "Person Only"],
    ["highlight_center", "Center Highlight"],
    ["thermal", "Thermal"]
  ];

  const PREDICTION_TOGGLES = [
    ["tracking", "Tracking"],
    ["tracers", "Tracers"],
    ["velocity", "Velocity"],
    ["prediction", "Prediction"]
  ];

  const STYLE_TOGGLES = [
    ["gay_mode", "RGB Mode"]
  ];

  const SLIDERS = [
    ["confidence_threshold", "Confidence Threshold", 0.05, 1.0, 0.05, (value) => `${Math.round(value * 100)}%`],
    ["line_thickness", "Line Thickness", 1, 6, 1, (value) => `${Math.round(value)} px`],
    ["face_zoom_scale", "Face Zoom Scale", 1.0, 4.0, 0.25, (value) => `${Number(value).toFixed(2)}x`],
    ["chams_opacity", "Chams Opacity", 0.0, 1.0, 0.05, (value) => `${Math.round(value * 100)}%`],
    ["face_blur_strength", "Face Blur Strength", 3, 35, 2, (value) => `${Math.round(value)}`]
  ];

  const COLORS = [
    ["color_primary", "Primary"],
    ["color_secondary", "Secondary"],
    ["color_boxes", "Boxes"],
    ["color_skeleton", "Skeleton"],
    ["color_head", "Head"],
    ["color_tracers", "Tracers"],
    ["color_velocity", "Velocity"],
    ["color_prediction", "Prediction"],
    ["color_face", "Face"],
    ["color_hand", "Hand"],
    ["color_text", "Text"],
    ["chams_color", "Chams"]
  ];

  const STORAGE_KEY = "securitycam.dashboard.workspace.v1";
  const STATUS_INTERVAL_MS = 1000;

  class DashboardApi {
    async fetchStatus(cameraId, signal) {
      const response = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/status`, { signal });
      return response.json();
    }

    async toggleYolo(cameraId) {
      const response = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/toggle_yolo`, {
        method: "POST"
      });
      return response.json();
    }

    async setModel(cameraId, modelName) {
      const response = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/set_model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelName })
      });
      return response.json();
    }

    async updateEsp(cameraId, payload) {
      const response = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/update_esp`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      return response.json();
    }
  }

  class DashboardApp {
    constructor(bootstrap) {
      this.api = new DashboardApi();
      this.cameras = Array.isArray(bootstrap.cameras) ? bootstrap.cameras : [];
      this.availableModels = Array.isArray(bootstrap.available_models) ? bootstrap.available_models : [];
      this.defaultCameraId = bootstrap.default_camera_id || (this.cameras[0] && this.cameras[0].id) || "";
      this.cameraMap = new Map(this.cameras.map((camera) => [camera.id, camera]));
      this.statusCache = new Map();
      this.pollTimer = null;
      this.statusAbortController = null;
      this.pendingPatchTimer = null;
      this.pendingPatch = {};
      this.pendingPatchCameraId = "";
      this.streamNonce = 0;
      this.clockTimer = null;
      this.isPageVisible = !document.hidden;

      this.state = {
        openCameraIds: [],
        activeCameraId: "",
        selectedModel: "",
        yoloEnabled: false,
        yoloAvailable: false,
        settings: {},
        stats: {}
      };

      this.elements = this.resolveElements();
    }

    resolveElements() {
      return {
        tabs: document.getElementById("camera-tabs"),
        directory: document.getElementById("camera-directory"),
        viewerTitle: document.getElementById("viewer-title"),
        stream: document.getElementById("stream"),
        streamPlaceholder: document.getElementById("stream-placeholder"),
        streamStatus: document.getElementById("stream-status"),
        streamFps: document.getElementById("stream-fps"),
        streamMessage: document.getElementById("stream-message"),
        activeCameraSummary: document.getElementById("active-camera-summary"),
        cameraName: document.getElementById("camera-name"),
        cameraAddress: document.getElementById("camera-address"),
        cameraPath: document.getElementById("camera-path"),
        toggleYoloButton: document.getElementById("btn-toggle-yolo"),
        modelSelect: document.getElementById("model-select"),
        clock: document.getElementById("clock"),
        classesList: document.getElementById("classes-list"),
        boxStyle: document.getElementById("sel-box-style"),
        visualToggleGrid: document.getElementById("visual-toggle-grid"),
        predictionToggleGrid: document.getElementById("prediction-toggle-grid"),
        styleToggleGrid: document.getElementById("style-toggle-grid"),
        sliderGroup: document.getElementById("slider-group"),
        colorGrid: document.getElementById("color-grid")
      };
    }

    init() {
      this.restoreWorkspace();
      this.buildControls();
      this.bindEvents();
      this.renderTabs();
      this.renderDirectory();
      this.setActiveCamera(this.state.activeCameraId || this.state.openCameraIds[0] || this.defaultCameraId);
      this.startClock();
      this.startPolling();
    }

    restoreWorkspace() {
      const stored = this.readStorage();
      const validIds = this.cameras.map((camera) => camera.id);
      const openCameraIds = Array.isArray(stored.openCameraIds)
        ? stored.openCameraIds.filter((cameraId) => validIds.includes(cameraId))
        : [];

      this.state.openCameraIds = openCameraIds.length > 0 ? openCameraIds : validIds.slice(0, Math.max(1, validIds.length));
      this.state.activeCameraId = validIds.includes(stored.activeCameraId) ? stored.activeCameraId : "";

      if (!this.state.activeCameraId || !this.state.openCameraIds.includes(this.state.activeCameraId)) {
        this.state.activeCameraId = this.state.openCameraIds[0] || this.defaultCameraId;
      }
    }

    readStorage() {
      try {
        return JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "{}");
      } catch (_) {
        return {};
      }
    }

    saveWorkspace() {
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
          openCameraIds: this.state.openCameraIds,
          activeCameraId: this.state.activeCameraId
        }));
      } catch (_) {
        // Ignore storage failures.
      }
    }

    bindEvents() {
      this.elements.toggleYoloButton.addEventListener("click", () => this.handleToggleYolo());
      this.elements.modelSelect.addEventListener("change", (event) => this.handleModelChange(event.target.value));
      this.elements.boxStyle.addEventListener("change", (event) => {
        this.state.settings.box_style = event.target.value;
        this.pushSettingPatch({ box_style: event.target.value }, true);
      });

      document.addEventListener("visibilitychange", () => {
        this.isPageVisible = !document.hidden;
        if (this.isPageVisible) {
          this.startPolling();
          this.refreshActiveCamera();
        } else {
          this.stopPolling();
        }
      });

      document.addEventListener("keydown", (event) => {
        const isCloseShortcut = (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "w";
        if (!isCloseShortcut) {
          return;
        }

        const activeElement = document.activeElement;
        if (activeElement && /input|select|textarea/i.test(activeElement.tagName)) {
          return;
        }

        event.preventDefault();
        this.closeCamera(this.state.activeCameraId);
      });

      this.elements.stream.addEventListener("load", () => {
        this.elements.streamPlaceholder.classList.add("is-hidden");
      });

      this.elements.stream.addEventListener("error", () => {
        this.elements.streamPlaceholder.classList.remove("is-hidden");
        this.elements.streamPlaceholder.textContent = "Stream unavailable. Check camera or RTSP settings.";
      });
    }

    buildControls() {
      this.buildToggleGroup(this.elements.visualToggleGrid, VISUAL_TOGGLES);
      this.buildToggleGroup(this.elements.predictionToggleGrid, PREDICTION_TOGGLES);
      this.buildToggleGroup(this.elements.styleToggleGrid, STYLE_TOGGLES);
      this.buildSliders();
      this.buildColors();
      this.renderModelOptions(this.availableModels);
    }

    buildToggleGroup(container, items) {
      items.forEach(([key, label]) => {
        const id = `toggle-${key}`;
        const wrapper = document.createElement("label");
        wrapper.className = "toggle-item";
        wrapper.innerHTML = `
          <span class="toggle-label">${label}</span>
          <span>
            <input id="${id}" class="toggle-input" type="checkbox">
            <span class="toggle-switch" aria-hidden="true"></span>
          </span>
        `;

        const input = wrapper.querySelector("input");
        input.addEventListener("change", () => {
          this.state.settings[key] = input.checked;
          this.pushSettingPatch({ [key]: input.checked }, true);
        });

        container.appendChild(wrapper);
      });
    }

    buildSliders() {
      SLIDERS.forEach(([key, label, min, max, step, formatter]) => {
        const row = document.createElement("div");
        row.className = "slider-row";
        row.innerHTML = `
          <div class="slider-header">
            <span>${label}</span>
            <span id="slider-value-${key}" class="slider-value">-</span>
          </div>
          <input id="slider-${key}" type="range" min="${min}" max="${max}" step="${step}">
        `;

        const input = row.querySelector("input");
        const valueElement = row.querySelector(".slider-value");

        input.addEventListener("input", () => {
          const value = Number(input.value);
          valueElement.textContent = formatter(value);
          this.state.settings[key] = value;
          this.pushSettingPatch({ [key]: this.normalizeSettingValue(key, value) }, false);
        });

        input.addEventListener("change", () => {
          const value = this.normalizeSettingValue(key, Number(input.value));
          this.state.settings[key] = value;
          valueElement.textContent = formatter(value);
          this.pushSettingPatch({ [key]: value }, true);
        });

        this.elements.sliderGroup.appendChild(row);
      });
    }

    buildColors() {
      COLORS.forEach(([key, label]) => {
        const wrapper = document.createElement("label");
        wrapper.className = "color-item";
        wrapper.innerHTML = `
          <span>${label}</span>
          <input id="color-${key}" type="color">
        `;

        const input = wrapper.querySelector("input");
        input.addEventListener("input", () => {
          this.state.settings[key] = input.value;
          this.pushSettingPatch({ [key]: input.value }, false);
        });

        input.addEventListener("change", () => {
          this.state.settings[key] = input.value;
          this.pushSettingPatch({ [key]: input.value }, true);
        });

        this.elements.colorGrid.appendChild(wrapper);
      });
    }

    normalizeSettingValue(key, value) {
      if (key === "line_thickness" || key === "face_blur_strength") {
        return Math.round(value);
      }
      return value;
    }

    pushSettingPatch(patch, immediate) {
      const cameraId = this.state.activeCameraId;
      if (!cameraId) {
        return;
      }

      if (this.pendingPatchCameraId && this.pendingPatchCameraId !== cameraId) {
        this.flushSettingPatch();
      }

      this.pendingPatchCameraId = cameraId;
      Object.assign(this.pendingPatch, patch);

      if (immediate) {
        this.flushSettingPatch();
        return;
      }

      window.clearTimeout(this.pendingPatchTimer);
      this.pendingPatchTimer = window.setTimeout(() => this.flushSettingPatch(), 180);
    }

    async flushSettingPatch() {
      const cameraId = this.pendingPatchCameraId;
      const payload = this.pendingPatch;
      this.pendingPatch = {};
      this.pendingPatchCameraId = "";
      window.clearTimeout(this.pendingPatchTimer);

      if (!cameraId || Object.keys(payload).length === 0) {
        return;
      }

      try {
        await this.api.updateEsp(cameraId, payload);
      } catch (_) {
        // Leave the last known UI state in place and refresh on next poll.
      }
    }

    renderTabs() {
      this.elements.tabs.innerHTML = "";

      this.state.openCameraIds.forEach((cameraId) => {
        const camera = this.cameraMap.get(cameraId);
        if (!camera) {
          return;
        }

        const tab = document.createElement("div");
        tab.className = `camera-tab${cameraId === this.state.activeCameraId ? " is-active" : ""}`;
        tab.dataset.cameraId = cameraId;
        tab.innerHTML = `
          <button type="button" class="camera-tab-main" aria-current="${cameraId === this.state.activeCameraId ? "page" : "false"}">
            <span class="camera-tab-label">${camera.label}</span>
          </button>
          <button type="button" class="camera-tab-close" aria-label="Close ${camera.label} tab">&times;</button>
        `;

        tab.querySelector(".camera-tab-main").addEventListener("click", () => this.setActiveCamera(cameraId));
        tab.querySelector(".camera-tab-close").addEventListener("click", (event) => {
          event.stopPropagation();
          this.closeCamera(cameraId);
        });

        this.elements.tabs.appendChild(tab);
      });
    }

    renderDirectory() {
      this.elements.directory.innerHTML = "";

      this.cameras.forEach((camera) => {
        const status = this.statusCache.get(camera.id) || {};
        const isOpen = this.state.openCameraIds.includes(camera.id);
        const isActive = this.state.activeCameraId === camera.id;

        const row = document.createElement("div");
        row.className = "camera-item";

        const stateLabel = status.stream_error
          ? status.stream_error
          : isActive
            ? "Open in workspace"
            : isOpen
              ? "Tab open"
              : "Available";

        row.innerHTML = `
          <div class="camera-item-main">
            <div class="camera-item-title">${camera.label}</div>
            <div class="camera-item-subtitle">${camera.ip || "No address"} | ${camera.stream_path || "No path"}</div>
            <div class="camera-item-state${status.yolo_enabled ? " is-active" : ""}">${stateLabel}</div>
          </div>
          <div>
            <button type="button" class="button button-ghost">${isOpen ? (isActive ? "Active" : "Focus") : "Open"}</button>
          </div>
        `;

        row.querySelector("button").addEventListener("click", () => {
          if (isOpen) {
            this.setActiveCamera(camera.id);
          } else {
            this.openCamera(camera.id);
          }
        });

        this.elements.directory.appendChild(row);
      });
    }

    openCamera(cameraId) {
      if (!this.cameraMap.has(cameraId)) {
        return;
      }

      if (!this.state.openCameraIds.includes(cameraId)) {
        this.state.openCameraIds.push(cameraId);
      }

      this.setActiveCamera(cameraId);
    }

    closeCamera(cameraId) {
      if (!cameraId || !this.state.openCameraIds.includes(cameraId)) {
        return;
      }

      const remaining = this.state.openCameraIds.filter((id) => id !== cameraId);
      if (remaining.length === 0) {
        const fallbackId = this.cameras.find((camera) => camera.id !== cameraId)?.id;
        if (fallbackId) {
          remaining.push(fallbackId);
        } else {
          return;
        }
      }

      this.state.openCameraIds = remaining;

      if (this.state.activeCameraId === cameraId) {
        this.state.activeCameraId = remaining[0];
      }

      this.saveWorkspace();
      this.renderTabs();
      this.renderDirectory();
      this.setActiveCamera(this.state.activeCameraId);
    }

    setActiveCamera(cameraId) {
      if (!cameraId || !this.cameraMap.has(cameraId)) {
        return;
      }

      if (!this.state.openCameraIds.includes(cameraId)) {
        this.state.openCameraIds.push(cameraId);
      }

      this.state.activeCameraId = cameraId;
      this.saveWorkspace();
      this.renderTabs();
      this.renderDirectory();
      this.renderCameraMetadata();
      this.switchStream(cameraId);
      this.refreshFromCache();
      this.refreshActiveCamera();
    }

    renderCameraMetadata() {
      const camera = this.cameraMap.get(this.state.activeCameraId);
      if (!camera) {
        return;
      }

      this.elements.viewerTitle.textContent = camera.label;
      this.elements.cameraName.textContent = camera.label;
      this.elements.cameraAddress.textContent = camera.ip || "Not configured";
      this.elements.cameraPath.textContent = camera.stream_path || "-";
    }

    switchStream(cameraId) {
      const camera = this.cameraMap.get(cameraId);
      const hasStream = camera && camera.has_stream;

      this.streamNonce += 1;
      this.elements.stream.src = "";

      if (!hasStream) {
        this.elements.streamPlaceholder.textContent = "No RTSP configuration found for this camera.";
        this.elements.streamPlaceholder.classList.remove("is-hidden");
        return;
      }

      this.elements.streamPlaceholder.textContent = "Connecting to camera stream...";
      this.elements.streamPlaceholder.classList.remove("is-hidden");
      this.elements.stream.src = `/video_feed/${encodeURIComponent(cameraId)}?v=${Date.now()}-${this.streamNonce}`;
    }

    refreshFromCache() {
      const status = this.statusCache.get(this.state.activeCameraId);
      if (!status) {
        this.resetStatusView();
        return;
      }

      this.applyStatus(status);
    }

    resetStatusView() {
      this.elements.streamStatus.textContent = "Standby";
      this.elements.streamStatus.className = "stream-status";
      this.elements.streamFps.textContent = "FPS -";
      this.elements.streamMessage.textContent = "";
      this.elements.activeCameraSummary.textContent = "Waiting for camera status";
      this.elements.toggleYoloButton.disabled = true;
      this.elements.modelSelect.disabled = false;
      this.renderRuntimeRows({});
      this.renderClasses([]);
    }

    async refreshActiveCamera() {
      const cameraId = this.state.activeCameraId;
      if (!cameraId || !this.isPageVisible) {
        return;
      }

      if (this.statusAbortController) {
        this.statusAbortController.abort();
      }

      this.statusAbortController = new AbortController();

      try {
        const status = await this.api.fetchStatus(cameraId, this.statusAbortController.signal);
        this.statusCache.set(cameraId, status);

        if (cameraId !== this.state.activeCameraId) {
          return;
        }

        this.applyStatus(status);
        this.renderDirectory();
      } catch (error) {
        if (error && error.name === "AbortError") {
          return;
        }

        this.elements.streamMessage.textContent = "Status request failed.";
        this.elements.streamStatus.textContent = "Unavailable";
        this.elements.streamStatus.className = "stream-status is-error";
      }
    }

    applyStatus(status) {
      this.state.yoloEnabled = Boolean(status.yolo_enabled);
      this.state.yoloAvailable = Boolean(status.yolo_available);
      this.state.selectedModel = status.yolo_model || this.availableModels[0] || "";
      this.state.settings = typeof status.esp_settings === "object" && status.esp_settings ? { ...status.esp_settings } : {};
      this.state.stats = {
        fps: Number(status.fps || 0),
        inference_fps: Number(status.inference_fps || 0),
        last_infer_ms: Number(status.last_infer_ms || 0),
        resolution: status.resolution || "-",
        num_detections: Number(status.num_detections || 0),
        faces: Number(status.faces || 0),
        classes: Array.isArray(status.classes) ? status.classes : [],
        total_frames: Number(status.total_frames || 0),
        device: status.device || "-",
        gpu: status.gpu || { name: "-", util: 0, mem_used: 0, mem_total: 0 }
      };

      if (Array.isArray(status.available_models)) {
        this.availableModels = status.available_models;
      }

      this.renderModelOptions(this.availableModels);
      this.syncControlsFromState();
      this.renderStreamState(status);
      this.renderRuntimeRows(this.state.stats);
      this.renderClasses(this.state.stats.classes);
    }

    renderModelOptions(models) {
      const selectedValue = this.state.selectedModel;
      const currentOptions = Array.from(this.elements.modelSelect.options).map((option) => option.value);
      const changed = currentOptions.length !== models.length || currentOptions.some((value, index) => value !== models[index]);

      if (changed) {
        this.elements.modelSelect.innerHTML = "";
        models.forEach((model) => {
          const option = document.createElement("option");
          option.value = model;
          option.textContent = model;
          this.elements.modelSelect.appendChild(option);
        });
      }

      if (selectedValue) {
        this.elements.modelSelect.value = selectedValue;
      }
    }

    syncControlsFromState() {
      const activeTag = document.activeElement ? document.activeElement.tagName : "";
      if (activeTag === "INPUT" || activeTag === "SELECT") {
        return;
      }

      [...VISUAL_TOGGLES, ...PREDICTION_TOGGLES, ...STYLE_TOGGLES].forEach(([key]) => {
        const input = document.getElementById(`toggle-${key}`);
        if (input) {
          input.checked = Boolean(this.state.settings[key]);
        }
      });

      SLIDERS.forEach(([key, , , , , formatter]) => {
        const input = document.getElementById(`slider-${key}`);
        const valueElement = document.getElementById(`slider-value-${key}`);
        if (!input || !valueElement) {
          return;
        }

        const value = this.state.settings[key];
        if (typeof value === "undefined") {
          return;
        }

        input.value = value;
        valueElement.textContent = formatter(Number(value));
      });

      COLORS.forEach(([key]) => {
        const input = document.getElementById(`color-${key}`);
        if (input && this.state.settings[key]) {
          input.value = this.state.settings[key];
        }
      });

      if (this.state.settings.box_style) {
        this.elements.boxStyle.value = this.state.settings.box_style;
      }
    }

    renderStreamState(status) {
      if (!status.yolo_available) {
        this.elements.streamStatus.textContent = "AI unavailable";
        this.elements.streamStatus.className = "stream-status is-error";
        this.elements.toggleYoloButton.textContent = "AI unavailable";
        this.elements.toggleYoloButton.className = "button button-danger";
        this.elements.toggleYoloButton.disabled = true;
      } else if (status.yolo_enabled) {
        this.elements.streamStatus.textContent = "AI active";
        this.elements.streamStatus.className = "stream-status is-active";
        this.elements.toggleYoloButton.textContent = "Stop AI";
        this.elements.toggleYoloButton.className = "button button-danger";
        this.elements.toggleYoloButton.disabled = false;
      } else {
        this.elements.streamStatus.textContent = "Standby";
        this.elements.streamStatus.className = "stream-status";
        this.elements.toggleYoloButton.textContent = "Start AI";
        this.elements.toggleYoloButton.className = "button button-primary";
        this.elements.toggleYoloButton.disabled = false;
      }

      this.elements.streamFps.textContent = `FPS ${this.formatNumber(this.state.stats.fps, 1)}`;
      this.elements.streamMessage.textContent = status.stream_error || "";

      if (status.stream_error) {
        this.elements.streamStatus.textContent = "Unavailable";
        this.elements.streamStatus.className = "stream-status is-error";
        this.elements.streamPlaceholder.textContent = status.stream_error;
        this.elements.streamPlaceholder.classList.remove("is-hidden");
      }

      const camera = status.camera || this.cameraMap.get(this.state.activeCameraId);
      const summaryParts = [
        camera && camera.label ? camera.label : "Camera",
        this.state.stats.resolution || "-",
        `FPS ${this.formatNumber(this.state.stats.fps, 1)}`
      ];
      this.elements.activeCameraSummary.textContent = summaryParts.join(" | ");
    }

    renderRuntimeRows(stats) {
      this.setText("stat-resolution", stats.resolution || "-");
      this.setText("stat-total-frames", this.formatInteger(stats.total_frames));
      this.setText("stat-fps", this.formatNumber(stats.fps, 1));
      this.setText("stat-infer-fps", this.formatNumber(stats.inference_fps, 1));
      this.setText("stat-infer-ms", stats.last_infer_ms ? `${this.formatNumber(stats.last_infer_ms, 1)} ms` : "-");
      this.setText("stat-detections", this.formatInteger(stats.num_detections));
      this.setText("stat-faces", this.formatInteger(stats.faces));
      this.setText("stat-yolo-status", this.state.yoloAvailable ? (this.state.yoloEnabled ? "active" : "standby") : "unavailable");
      this.setText("stat-model", this.state.selectedModel || "-");
      this.setText("stat-device", stats.device || "-");
      this.setText("stat-gpu-name", stats.gpu && stats.gpu.name ? stats.gpu.name : "-");
      this.setText("stat-gpu-util", stats.gpu && stats.gpu.util ? `${stats.gpu.util}%` : "-");
      this.setText(
        "stat-gpu-mem",
        stats.gpu && stats.gpu.mem_total
          ? `${this.formatInteger(stats.gpu.mem_used)} / ${this.formatInteger(stats.gpu.mem_total)} MB`
          : "-"
      );
    }

    renderClasses(classes) {
      this.elements.classesList.innerHTML = "";
      const items = Array.isArray(classes) && classes.length > 0 ? classes : ["none"];
      items.forEach((item) => {
        const li = document.createElement("li");
        li.className = "class-item";
        li.textContent = item;
        this.elements.classesList.appendChild(li);
      });
    }

    async handleToggleYolo() {
      if (!this.state.activeCameraId || !this.state.yoloAvailable) {
        return;
      }

      try {
        await this.api.toggleYolo(this.state.activeCameraId);
        await this.refreshActiveCamera();
      } catch (_) {
        this.elements.streamMessage.textContent = "Could not update AI state.";
      }
    }

    async handleModelChange(modelName) {
      if (!this.state.activeCameraId || !modelName) {
        return;
      }

      try {
        await this.api.setModel(this.state.activeCameraId, modelName);
        this.state.selectedModel = modelName;
        await this.refreshActiveCamera();
      } catch (_) {
        this.elements.streamMessage.textContent = "Could not change the model.";
      }
    }

    startClock() {
      const renderClock = () => {
        this.elements.clock.textContent = new Date().toLocaleTimeString("de-DE", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit"
        });
      };

      renderClock();
      this.clockTimer = window.setInterval(renderClock, 1000);
    }

    startPolling() {
      this.stopPolling();
      if (!this.isPageVisible) {
        return;
      }

      this.pollTimer = window.setInterval(() => this.refreshActiveCamera(), STATUS_INTERVAL_MS);
    }

    stopPolling() {
      window.clearInterval(this.pollTimer);
      this.pollTimer = null;
    }

    setText(id, value) {
      const element = document.getElementById(id);
      if (element) {
        element.textContent = value;
      }
    }

    formatNumber(value, digits) {
      return value ? Number(value).toFixed(digits) : "-";
    }

    formatInteger(value) {
      return Number.isFinite(value) && value >= 0 ? String(Math.round(value)) : "-";
    }
  }

  function readBootstrap() {
    const element = document.getElementById("bootstrap-data");
    if (!element) {
      return {};
    }

    try {
      return JSON.parse(element.textContent || "{}");
    } catch (_) {
      return {};
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const app = new DashboardApp(readBootstrap());
    app.init();
  });
})();
