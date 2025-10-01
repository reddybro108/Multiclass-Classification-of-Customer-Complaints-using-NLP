from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

class InputText(BaseModel):
    text: str

app = FastAPI()

id2department = {
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

num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your pretrained and fine-tuned DistilBERT model and tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("./saved_bert_model")
model = DistilBertForSequenceClassification.from_pretrained("./saved_bert_model")

model.to(device)
model.eval()

@app.post("/predict")
async def predict(input: InputText):
    inputs = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    label = id2department.get(pred_idx, str(pred_idx))

    return {
        "label_id": pred_idx,
        "label": label,
        "confidence": confidence,
    }
