import torch

def load_partial_weights(model, weight_path, device):
    """
    Loads weights except final classifier layer.
    Used for cascade fine-tuning.
    """
    state_dict = torch.load(weight_path, map_location=device)

    model_dict = model.state_dict()

    filtered_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    print("✅ Loaded backbone weights for fine-tuning")
    return model
