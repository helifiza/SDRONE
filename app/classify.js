console.log("classify.js loaded");
const API_link = "http://localhost:8000";

const fileInput     = document.querySelector(".drop_box input[type='file']");
const dropBox       = document.querySelector(".drop_box");
const uploadBtn     = document.querySelector(".upload_btn");
const result        = document.querySelector(".result");
const loadingSignal = document.querySelector(".loading_signal");

dropBox.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropBox.classList.add("drag-over");
});

dropBox.addEventListener("dragleave", () => dropBox.classList.remove("drag-over"));

dropBox.addEventListener("drop", (e) => {
    e.preventDefault();
    dropBox.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) assignFile(file);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) assignFile(fileInput.files[0]);
});

function assignFile(file) {
    if (!file.name.toLowerCase().endsWith(".mat")) {
        showError("⚠️ Hệ thống chỉ nhận file .mat");
        return;
    }
    fileInput._selectedFile = file;
    dropBox.querySelector("i").style.display = "none";

    let nameTag = dropBox.querySelector(".file-name");
    if (!nameTag) {
        nameTag = document.createElement("span");
        nameTag.className = "file-name";
        dropBox.appendChild(nameTag);
    }
    nameTag.textContent = file.name;

    // Reset về trạng thái ban đầu
    result.innerHTML = "";
    result.style.display = "none";
    loadingSignal.style.display = "flex";
}

uploadBtn.addEventListener("click", async () => {
    const file = fileInput._selectedFile || fileInput.files[0];

    if (!file) {
        showError("⚠️ Vui lòng chọn file .mat trước khi upload");
        return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${API_link}/predict/mat`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        renderResult(data);

    } catch (err) {
        showError(`Lỗi: ${err.message}`);
    } finally {
        setLoading(false);
    }
});

function renderResult(d) {
    // Ẩn loading_signal, hiện result
    loadingSignal.style.display = "none";
    result.style.display = "block";
    result.innerHTML = "";

    const card = document.createElement("div");
    card.className = "result-card " + (d.drone_detected ? "detected" : "clear");

    if (!d.drone_detected) {
        const title = document.createElement("p");
        title.className = "result-title";
        title.textContent = "Hệ thống không phát hiện drone";

        const icon = document.createElement("span");
        icon.className = "result-icon";
        icon.textContent = "✅";

        card.appendChild(title);
        card.appendChild(icon);
        result.appendChild(card);
        return;
    }

    const title = document.createElement("p");
    title.className = "result-title";
    title.innerHTML = "&#128680; Hệ thống phát hiện có drone";
    card.appendChild(title);

    const entries  = Object.entries(d.type_detail).filter(function(e) { return e[0] !== "Unknown"; });
    const showList = entries.length > 0 ? entries : [["Unknown", { frequency: 1 }]];

    showList.forEach(function(entry) {
        const type = entry[0];
        const pct  = (entry[1].frequency * 100).toFixed(1) + "%";

        const row = document.createElement("div");
        row.className = "drone-row";

        const nameBox = document.createElement("div");
        nameBox.className = "drone-name";
        nameBox.textContent = type;

        const pctBox = document.createElement("div");
        pctBox.className = "drone-pct";
        pctBox.textContent = pct;

        row.appendChild(nameBox);
        row.appendChild(pctBox);
        card.appendChild(row);
    });

    result.appendChild(card);
}

function showError(msg) {
    loadingSignal.style.display = "none";
    result.style.display = "block";
    result.innerHTML = '<p class="error-msg">' + msg + '</p>';
}

function setLoading(on) {
    uploadBtn.disabled = on;
    uploadBtn.textContent = on ? "Đang xử lý..." : "Upload";
    if (on) {
        loadingSignal.style.display = "flex";
        result.style.display = "none";
        result.innerHTML = "";
    }
}