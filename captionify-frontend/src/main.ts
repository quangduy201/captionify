import './style.css'
import addImageIcon from '/assets/add_image.png'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="container">
    <h1>Captionify</h1>
    <p>An AI Image Caption Generator</p>

    <div class="main-content">
      <label for="fileElem" hidden></label>
      <div class="upload-box" id="drop-area">
        <input type="file" id="fileElem" accept="image/*" title="Image" hidden />
        <div class="placeholder" id="placeholder">
          <img src="${addImageIcon}" alt="Add image" width="72"/>
          <p>Drag & drop an image here, or <span class="browse">browse</span></p>
        </div>
        <img id="preview" class="preview-img" alt="" src=""/>
      </div>

      <button id="startBtn" type="button">Generate caption</button>

      <div class="caption-box">
        <label for="caption" hidden></label>
        <textarea id="caption" title="Generated Caption" readonly></textarea>
      </div>
    </div>
  </div>
`

const API_URL = import.meta.env.VITE_API_URL;
const speed = 70; // milliseconds

const dropArea = document.getElementById("drop-area")!;
const fileInput = document.getElementById("fileElem") as HTMLInputElement;
const browseText = document.querySelector(".browse")!;
const previewImg = document.getElementById("preview") as HTMLImageElement;
const placeholder = document.getElementById("placeholder")!;
const startBtn = document.getElementById("startBtn") as HTMLButtonElement;
const captionTextarea = document.getElementById("caption") as HTMLTextAreaElement;

let selectedFile: File | null = null;

browseText.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropArea.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  selectedFile = fileInput.files?.[0] || null;
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
  selectedFile = e.dataTransfer?.files?.[0] || null;
  previewAndShow(selectedFile);
});

function previewAndShow(file: File | null) {
  if (file && file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = () => {
      previewImg.src = reader.result as string;
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

    const response = await fetch(`${API_URL}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Upload failed.");

    const data = await response.json();
    showTypingEffect(data.caption);
  } catch (err: any) {
    captionTextarea.value = `Error: ${err.message}`;
  } finally {
    startBtn.disabled = false;
  }
});

function showTypingEffect(text: string) {
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
