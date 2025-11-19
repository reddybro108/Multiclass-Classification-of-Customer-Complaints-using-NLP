from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from contextlib import asynccontextmanager

# Import settings from the new config file
from app.config import get_settings

# Create a single settings instance
settings = get_settings()

# ---------- Lifespan Management (Model Loading) ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and tokenizer on startup
    print("INFO:     Loading model and tokenizer...")
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(settings.SAVED_MODEL_PATH, local_files_only=True)
        model = DistilBertForSequenceClassification.from_pretrained(settings.SAVED_MODEL_PATH, local_files_only=True)
        print(f"✅ Loaded model & tokenizer from local path: {settings.SAVED_MODEL_PATH}")
    except Exception:
        print(f"⚠️ Local model not found at {settings.SAVED_MODEL_PATH}. Falling back to HuggingFace hub.")
        tokenizer = DistilBertTokenizerFast.from_pretrained(settings.BASE_MODEL_NAME)
        model = DistilBertForSequenceClassification.from_pretrained(settings.BASE_MODEL_NAME)

    # Assign to app state
    app.state.tokenizer = tokenizer
    app.state.model = model.to(settings.DEVICE)
    app.state.device = settings.DEVICE
    app.state.model.eval()
    print("INFO:     Model and tokenizer loaded and assigned to app state.")
    
    yield
    
    # Clean up resources if needed
    print("INFO:     Application shutdown.")


# ---------- FastAPI App ----------
app = FastAPI(
    title="Complaint Classifier API",
    version="1.0",
    lifespan=lifespan  # Use the lifespan manager
)

class InputText(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Complaint Classification API is running."}

@app.post("/predict")
async def predict(request: Request, input: InputText):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Access model, tokenizer, and device from app state
    tokenizer = request.app.state.tokenizer
    model = request.app.state.model
    device = request.app.state.device

    try:
        # Tokenization
        inputs = tokenizer(
            input.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=settings.MAX_LEN,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        label = settings.ID2DEPARTMENT.get(pred_idx, str(pred_idx))

        return {
            "label_id": pred_idx,
            "label": label,
            "confidence": round(confidence, 4),
        }

    except Exception as e:
        # Log the error for debugging
        print(f"ERROR:    Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ---------- Run with Uvicorn ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS
    )