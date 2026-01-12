from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

def animate_predictions(input_seq, target_seq, pred_seq, title="Анимация"):
    """Создание анимации предсказаний"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    im1 = axes[0].imshow(input_seq[0], cmap='gray')
    im2 = axes[1].imshow(target_seq[0], cmap='gray')
    im3 = axes[2].imshow(pred_seq[0], cmap='gray')

    axes[0].set_title("Вход")
    axes[1].set_title("Цель")
    axes[2].set_title("Предсказание")

    for ax in axes:
        ax.axis('off')

    def update(frame):
        if frame < len(input_seq):
            im1.set_data(input_seq[frame])
        if frame < len(target_seq):
            im2.set_data(target_seq[frame])
        if frame < len(pred_seq):
            im3.set_data(pred_seq[frame])
        return im1, im2, im3

    anim = FuncAnimation(fig, update, frames=max(len(input_seq), len(target_seq), len(pred_seq)),
                        interval=200, blit=True)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()


    html = anim.to_jshtml()
    plt.close(fig)
    return HTML(html)



def visualize_predictions_comparison(model, test_input, test_target,
                                     model_name="Модель", epoch=1,
                                     num_prediction_steps=None, figsize=(15, 9)):
    """Функция визуализации предсказаний модели"""

    model.eval()
    with torch.no_grad():
        if num_prediction_steps is not None:
            test_pred = model(test_input, num_prediction_steps=num_prediction_steps)
        else:
            test_pred = model(test_input)

        input_np = test_input[:, 0].cpu().numpy()
        target_np = test_target[:, 0].cpu().numpy()
        pred_np = test_pred[:, 0].cpu().numpy()

        fig, axes = plt.subplots(3, 5, figsize=figsize)

        # Отображаем первые 5 кадров
        for i in range(5):
            if i < len(input_np):
                axes[0, i].imshow(input_np[i], cmap='gray')
            else:
                axes[0, i].imshow(np.zeros_like(input_np[0]), cmap='gray')
            axes[0, i].set_title(f"Вход {i + 1}")
            axes[0, i].axis('off')

            if i < len(target_np):
                axes[1, i].imshow(target_np[i], cmap='gray')
            else:
                axes[1, i].imshow(np.zeros_like(target_np[0]), cmap='gray')
            axes[1, i].set_title(f"Цель {i + 1}")
            axes[1, i].axis('off')

            if i < len(pred_np):
                axes[2, i].imshow(pred_np[i], cmap='gray')
            else:
                axes[2, i].imshow(np.zeros_like(pred_np[0]), cmap='gray')
            axes[2, i].set_title(f"Предсказание {i + 1}")
            axes[2, i].axis('off')

        plt.suptitle(f"{model_name} - Эпоха {epoch}", fontsize=14)
        plt.tight_layout()
        plt.show()

    return test_pred


def plot_losses(train_losses, test_losses, title="График потерь", window=5):
    """Построение графиков train и test потерь по эпохам"""
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='Train Loss')
    plt.plot(epochs, test_losses, 'r-', linewidth=2, marker='s', markersize=4, label='Test Loss')
    plt.fill_between(epochs, train_losses, test_losses, alpha=0.1, color='gray')

    diff = [t - tr for tr, t in zip(train_losses, test_losses)]
    plt.plot(epochs, diff, 'g--', alpha=0.7, linewidth=1, label='Разница (Test - Train)')

    plt.axhline(y=0, color='k', linestyle=':', alpha=0.3)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Эпоха", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.show()