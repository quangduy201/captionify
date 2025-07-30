const speed = 70; // milliseconds

const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("fileElem");
const browseText = document.querySelector(".browse");
const previewImg = document.getElementById("preview");
const placeholder = document.getElementById("placeholder");
const removeBtn = document.getElementById("removeBtn");
const startBtn = document.getElementById("startBtn");
const captionTextarea = document.getElementById("caption");

let selectedFile = null; // lưu file đã chọn

browseText.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropArea.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  selectedFile = e.target.files[0];
  previewAndShow(selectedFile);
});

dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.classList.add("dragging");
});

dropArea.addEventListener("dragleave", () => {
  dropArea.classList.remove("dragging");
});

dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  dropArea.classList.remove("dragging");
  selectedFile = e.dataTransfer.files[0];
  previewAndShow(selectedFile);
});

function previewAndShow(file) {
  if (file && file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = () => {
      previewImg.src = reader.result;
      previewImg.style.display = "block";
      placeholder.style.display = "none";
      dropArea.classList.add("has-preview");
    };
    reader.readAsDataURL(file);
  }
}

startBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    alert("Please upload an image first!");
    return;
  }

  captionTextarea.value = "Processing image...";
  startBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("http://localhost:8000/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Response error.");

    const data = await response.json();
    console.log(data);
    console.log(data.caption);
    showTypingEffect(data.caption);
  } catch (err) {
    captionTextarea.value = "Error: " + err.message;
  } finally {
    startBtn.disabled = false;
  }
});

function showTypingEffect(text) {
  captionTextarea.value = "";
  let index = 0;

  const interval = setInterval(() => {
    if (index < text.length) {
      captionTextarea.value += text.charAt(index);
      index++;
    } else {
      clearInterval(interval);
    }
  }, speed);
}
