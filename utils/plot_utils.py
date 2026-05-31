import os

import matplotlib.pyplot as plt


def plot_training_curves(
    train_loss_list,
    train_acc_list,
    test_loss_list,
    test_acc_list,
    save_path
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(train_loss_list) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.plot(epochs, test_loss_list, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_list, label="Train Accuracy")
    plt.plot(epochs, test_acc_list, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"训练曲线已保存到: {save_path}")
