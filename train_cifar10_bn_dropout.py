import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cifar_cnn_bn_dropout import CIFAR10CNN_BN_Dropout
from utils.train_utils import train_one_epoch, evaluate
from utils.checkpoint_utils import save_checkpoint
from utils.plot_utils import plot_training_curves


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

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
    print("图像形状:", images.shape)
    print("标签形状:", labels.shape)

    model = CIFAR10CNN_BN_Dropout(
        num_classes=10
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 10
    best_test_acc = 0.0
    save_path = "checkpoints/best_cifar10_bn_dropout_cnn.pth"

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

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

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(model, save_path)
            print(f"发现更好的 BN + Dropout 模型，当前最佳 Test Acc: {best_test_acc:.4f}")

    plot_training_curves(
        train_loss_list=train_loss_list,
        train_acc_list=train_acc_list,
        test_loss_list=test_loss_list,
        test_acc_list=test_acc_list,
        save_path="results/cifar10_bn_dropout_curves.png"
    )


if __name__ == "__main__":
    main()
