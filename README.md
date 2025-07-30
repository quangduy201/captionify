# Image Captioning Web App

This is a simple **image captioning web application** powered by a deep learning
model trained on datasets like **Flickr8k**. Users can upload an image, and the
model will generate a natural language description of the image.

> The project uses a **FastAPI backend** (with PyTorch model inference) and a responsive **HTML/CSS/JS frontend**.

## Features

- Upload any image by drag-and-dropping or browsing files
- Generate English captions using a trained neural network
- Live typing effect for displaying generated captions
- Trained on Flickr8k dataset with CNN+RNN architecture
- `POST /reload-model` route to reload the latest model from Kaggle without restarting


## Setup

### 1. Clone the repository:
```shell
git clone https://github.com/quangduy201/image_captioning.git
cd image_captioning
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
- Automatically download the latest model from [`quangduy201/image-captioning/pyTorch/checkpoint`](https://www.kaggle.com/models/quangduy201/image-captioning) Kaggle Models
- Load model + vocabulary into memory
- Start FastAPI on [`http://localhost:8000`](http://localhost:8000)


### 5. Access the application:
Open a web browser and go to [`http://localhost:8000`](http://localhost:8000) to access the application.


## Train custom model

If you want to train a custom image-captioning model, you can use the provided Kaggle notebook [here](https://www.kaggle.com/code/quangduy201/image-captioning-pytorch)
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
