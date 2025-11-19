from pydantic_settings import BaseSettings
from functools import lru_cache
import torch

class Settings(BaseSettings):
    """
    Application settings managed by Pydantic.
    Reads from environment variables and .env files.
    """
    # ---------- Inference Settings ----------
    # Path to the saved model for inference
    SAVED_MODEL_PATH: str = "./saved_bert_model"
    # Set device automatically
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- Model & Training Settings ----------
    # Base model for training (if saved_model_path is not found)
    BASE_MODEL_NAME: str = "distilbert-base-uncased"
    NUM_CLASSES: int = 10
    MAX_LEN: int = 128
    TRAIN_BATCH_SIZE: int = 16
    EVAL_BATCH_SIZE: int = 16
    EPOCHS: int = 3
    LR: float = 2e-5

    # ---------- API Server Settings ----------
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    WORKERS: int = 1
    RELOAD: bool = True

    # Department mapping
    ID2DEPARTMENT: dict = {
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

    class Config:
        # This tells pydantic to load variables from a .env file
        env_file = ".env"
        env_file_encoding = "utf-8"

# Use lru_cache to create a singleton-like settings object
@lru_cache
def get_settings():
    return Settings()