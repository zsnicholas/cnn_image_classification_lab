import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.simple_cnn import SimpleCNN
from utils.train_utils import train_one_epoch, evaluate

from utils.checkpoint_utils import save_checkpoint


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    images, labels = next(iter(train_loader))
    print("图像形状：", images.shape)
    print("标签形状：", labels.shape)

    model = SimpleCNN(
        in_channels=1,
        num_classes=10
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 3
    best_test_acc = 0.0
    save_path = "checkpoints/best_mnist_cnn.pth"

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            loss_fn,
            device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, "
            f"Test Acc: {test_acc:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc =test_acc
            save_checkpoint(model, save_path)
            print(f"发现更好的 MNIST 模型，当前最佳 Test Acc：{best_test_acc:.4f}")


if __name__ == "__main__":
    main()
