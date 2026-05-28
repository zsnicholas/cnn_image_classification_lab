import torch
from PIL import Image, ImageOps
from torchvision import transforms

from models.simple_cnn import SimpleCNN
from utils.checkpoint_utils import load_checkpoint


def preprocess_mnist_image(image_path):
    image = Image.open(image_path).convert("L")

    # 找到非黑色区域，也就是数字区域
    bbox = image.getbbox()

    if bbox is not None:
        image = image.crop(bbox)

    # 保持比例，加黑边补成正方形
    image = ImageOps.pad(
        image,
        size=(28, 28),
        color=0,
        centering=(0.5, 0.5)
    )

    transform = transforms.ToTensor()
    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    image_tensor = preprocess_mnist_image(image_path)

    print("处理后图片形状：", image_tensor.shape)

    image_tensor = image_tensor.to(device)

    model = SimpleCNN(in_channels=1, num_classes=10)

    model = load_checkpoint(
        model,
        "checkpoints/best_mnist_cnn.pth",
        device
    )

    model.eval()

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

    print("预测数字：", pred.item())
    print("预测概率：", probs[0, pred.item()].item())

    print("全部类别概率：")
    for i in range(10):
        print(f"数字{i}: {probs[0, i].item():.4f}")


if __name__ == "__main__":
    image_path = "test_images/mnist_test.png"
    predict_image(image_path)