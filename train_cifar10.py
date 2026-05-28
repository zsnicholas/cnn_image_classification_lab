import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cifar_cnn import CIFAR10CNN
from utils.train_utils import train_one_epoch, evaluate
from utils.checkpoint_utils import save_checkpoint


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    train_transform = transforms.Compose([
        transforms.RandomCrop(
            size=32,
            padding=4
        ),
        transforms.RandomHorizontalFlip(
            p=0.5
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    images, labels = next(iter(train_loader))

    print("图像形状：", images.shape)
    print("标签形状：", labels.shape)
    print("前 10 个标签：", labels[:10])

    for i in range(10):
        label_id = labels[i].item()
        print(f"样本{i}: 标签编号={label_id}, 类别名称={class_names[label_id]}")

    model = CIFAR10CNN(
        num_classes=10
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 10
    best_test_acc = 0.0
    save_path = "checkpoints/best_cifar10_aug_cnn.pth"

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
            best_test_acc = test_acc
            save_checkpoint(model, save_path)
            print(f"发现更好的 CIFAR-10 增强模型，当前最佳 Test Acc: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()