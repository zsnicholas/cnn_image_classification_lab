import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cifar_cnn import CIFAR10CNN
from utils.checkpoint_utils import load_checkpoint


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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    test_dataset = datasets.CIFAR10(
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

    model = CIFAR10CNN(
        num_classes=10
    )

    model = load_checkpoint(
        model,
        "checkpoints/best_cifar10_cnn.pth",
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

    true_id = label.item()
    pred_id = pred.item()

    print("真实标签编号：", true_id)
    print("真实类别名称：", class_names[true_id])
    print("预测标签编号：", pred_id)
    print("预测类别名称：", class_names[pred_id])
    print("预测概率：", probs[0, pred_id].item())

    print("全部类别概率：")
    for i in range(10):
        print(f"{class_names[i]}: {probs[0, i].item():.4f}")


if __name__ == "__main__":
    main()

