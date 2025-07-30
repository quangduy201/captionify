# Captionify - An AI Image Caption Generator

**Captionify** is a simple web application that generates English captions from images using a deep learning model.
It combines a **FastAPI backend** (for model inference) and a responsive **HTML/CSS/JS frontend**.

> The model is trained on datasets like **Flickr8k** with a **CNN + RNN architecture**.

## Features

- Upload images drag & drop or file browser.
- Generate descriptive English captions from your images.
- Live typing effect for generated captions.
- Trained on Flickr8k dataset with a custom PyTorch model.
- API endpoint `POST /reload-model` to dynamically reload the latest model from Kaggle.


## Setup

### 1. Clone the repository:
```shell
git clone https://github.com/quangduy201/captionify.git
cd captionify
```

### 2. Create and activate a virtual environment:
```shell
# Create a virtual environment named '.venv'
python -m venv .venv

# Activate the virtual environment
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```
   
### 3. Install dependencies:
```shell
pip install -r requirements.txt
```
    
### 4. Run:
```shell
uvicorn run:app --reload
```

This will:
- Automatically download the latest model from [`Kaggle Model Hub`](https://www.kaggle.com/models/quangduy201/image-captioning).
- Load model + vocabulary into memory.
- Start FastAPI on [`http://localhost:8000`](http://localhost:8000)


### 5. Access the application:
Open a web browser and go to [`http://localhost:8000`](http://localhost:8000) to access the application.


## Train your own model

You can train a custom captioning model using the provided Kaggle notebook [here](https://www.kaggle.com/code/quangduy201/image-captioning-pytorch)
### 1. Open the notebook:
Visit the provided Kaggle link and create a copy of the notebook.

### 2. Choose your suitable dataset:
You can choose which dataset which is the most suitable for your model.
The default dataset is [flickr8k](https://www.kaggle.com/datasets/quangduy201/flickr8k).

### 2. Settings for the notebook's session
You should train the model using GPU T4 x2 or GPU P100

### 3. Run the notebook:
Simply press `Run all` in the notebook to train your custom model.

### 4. Download the trained model:
After training, download the trained model checkpoint (`/kaggle/working/training/output/checkpoint.pth.tar`).

### 5. Place the trained model:
Place the downloaded `checkpoint.pth.tar` file in the `training/output` directory of the repository.

### 6. Improve the trained model (Optional):
If you want to improve the trained model, you can upload your current checkpoint of the model
and use it as an Input of the notebook.


## Dependencies

- fastapi
- uvicorn
- torch
- torchvision
- spacy
- tqdm
- Pillow
- python-multipart
- tensorboard
- kagglehub
