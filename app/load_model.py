from transformers import DistilBertConfig, DistilBertForSequenceClassification
import torch

def load_model_with_custom_head(checkpoint_path, num_classes=10, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model config with the desired number of labels
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased', num_labels=num_classes)

    # Initialize new model with this config
    model = DistilBertForSequenceClassification(config)

    # Load state dictionary from old checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Remove old classification head weights from state dict
    keys_to_remove = [k for k in state_dict.keys() if 'classifier' in k]
    for key in keys_to_remove:
        state_dict.pop(key)

    # Load remaining weights into the model (backbone only)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model
