import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.simple_cnn import SimpleCNN
from utils.checkpoint_utils import load_checkpoint


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True
    )

    model = SimpleCNN(
        in_channels=1,
        num_classes=10
    )

    mmodel = load_checkpoint(
        model,
        "checkpoints/best_mnist_cnn.pth",
        device
    )

    model.eval()

    image, label = next(iter(test_loader))

    image = image.to(device)
    label = label.to(device)

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

    print("真实标签：", label.item())
    print("预测标签：", pred.item())
    print("预测概率：", probs[0, pred.item()].item())

    print("全部类别概率：")
    for i in range(10):
        print(f"数字{i}:{probs[0, i].item():.4f}")


if __name__ == "__main__":
    main()
