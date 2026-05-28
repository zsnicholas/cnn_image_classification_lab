import torch
from PIL import Image
from torchvision import transforms

from models.cifar_cnn import CIFAR10CNN
from utils.checkpoint_utils import load_checkpoint


def predict_image(image_path):
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
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    image = Image.open(image_path)
    image_tensor = transform(image)

    print("加 batch 维度前：", image_tensor.shape)

    image_tensor = image_tensor.unsqueeze(0)

    print("加 batch 维度后：", image_tensor.shape)

    image_tensor = image_tensor.to(device)

    model = CIFAR10CNN(
        num_classes=10
    )

    model = load_checkpoint(
        model,
        "checkpoints/best_cifar10_cnn.pth",
        device
    )

    model.eval()

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

    pred_id = pred.item()
    pred_name = class_names[pred_id]
    pred_prob = probs[0, pred_id].item()

    print("预测类别编号：", pred_id)
    print("预测类别名称：", pred_name)
    print("预测概率：", pred_prob)

    print("全部类别概率：")
    for i in range(10):
        print(f"{class_names[i]}: {probs[0, i].item():.4f}")


if __name__ == "__main__":
    image_path = "test_images/cifar_test.jpg"
    predict_image(image_path)