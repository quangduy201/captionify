import gc

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

from app.model.load_model import load_model_and_vocab_from_kaggle
from app.services.image_captioner import generate_caption

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
def root():
    gc.collect()
    return JSONResponse(status_code=200, content={"message": "Hello World!"})


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    caption = generate_caption(file.file)
    return JSONResponse(status_code=200, content={"caption": caption})


@router.post("/reload-model")
def reload_model():
    try:
        load_model_and_vocab_from_kaggle(force_download=True)
        return {"message": "✅ Model reloaded successfully."}
    except Exception as e:
        return {"error": str(e)}
