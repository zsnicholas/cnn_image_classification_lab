import os
import torch


def save_checkpoint(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到：{save_path}")


def load_checkpoint(model, load_path, device):
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"模型已从{load_path}加载")
    return model
