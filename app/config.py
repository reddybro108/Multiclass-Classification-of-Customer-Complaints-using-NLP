class Config:
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 16
    EPOCHS = 3
    LR = 2e-5
    DATA_PATH = "data/processed/train.csv"
    MODEL_SAVE_PATH = "models/bert_classifier"
