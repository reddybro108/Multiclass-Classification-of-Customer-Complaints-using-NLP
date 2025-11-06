from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# ---------- Config ----------
MODEL_PATH = "./saved_bert_model"  # local folder or HuggingFace model name
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapping of prediction IDs to department labels
ID2DEPARTMENT = {
    0: "Credit Reporting, Credit Repair, Consumer Reports",
    1: "Debt Collection",
    2: "Mortgage",
    3: "Credit Card or Prepaid Card",
    4: "Checking or Savings Account",
    5: "Student Loan",
    6: "Consumer Loan",
    7: "Money Transfer or Virtual Currency",
    8: "Vehicle Loan or Lease",
    9: "Other Financial Service"
}

# ---------- Model & Tokenizer Load ----------
def load_model_and_tokenizer(model_path: str):
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path, local_files_only=True)
        model = DistilBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        print(f"✅ Loaded model & tokenizer from local path: {model_path}")
    except Exception:
        print(f"⚠️ Local model not found or incomplete at {model_path}. Falling back to HuggingFace hub.")
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer(MODEL_PATH)
model.to(DEVICE)
model.eval()

# ---------- FastAPI ----------
app = FastAPI(title="Complaint Classifier API", version="1.0")

class InputText(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Complaint Classification API is running."}

@app.post("/predict")
async def predict(input: InputText):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        # Tokenization
        inputs = tokenizer(
            input.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        label = ID2DEPARTMENT.get(pred_idx, str(pred_idx))

        return {
            "label_id": pred_idx,
            "label": label,
            "confidence": round(confidence, 4),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ---------- Run with Uvicorn ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, workers=1)
