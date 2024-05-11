# Image Captioning Django Application

This Django application utilizes a trained image captioning model
to generate descriptions for images uploaded by users.
It provides an easy-to-use interface for uploading images,
processing them with the pre-trained model,
and displaying the generated captions.

## Features

- Upload images to receive automatically generated captions.
- Utilizes a pre-trained image captioning model for caption generation.
- Option to train a custom image captioning model using a provided Kaggle notebook.


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
    
### 4. Download the dataset & pre-trained model:
- Download the default dataset [flickr8k](https://www.kaggle.com/datasets/quangduy201/flickr8k).
- Place the downloaded `flickr8k` dataset in the [`data`](data) directory of the repository.
- Download the pre-trained model checkpoint file from Kaggle [here](https://www.kaggle.com/models/quangduy201/image-captioning)
- Place the downloaded `checkpoint.pth.tar` file in the [`resources`](resources) directory of the repository.

### 5. Run the Django application:
```shell
python manage.py runserver
```

### 6. Access the application:
Open a web browser and go to [`http://localhost:8000`](http://localhost:8000) to access the application.


## Train custom model

If you want to train a custom image captioning model, you can use the provided Kaggle notebook [here](https://www.kaggle.com/code/quangduy201/image-captioning-pytorch)
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
After training, download the trained model checkpoint (/kaggle/working/checkpoint.pth.tar).

### 5. Place the trained model:
Place the downloaded `checkpoint.pth.tar` file in the [`resources`](resources) directory of the repository.

### 6. Improve the trained model (Optional):
If you want to improve the trained model, you can upload your current checkpoint of the model
and use it as an Input of the notebook.


## Project structure


## Dependencies

- Django
- torch
- torchvision
- spacy
- tqdm