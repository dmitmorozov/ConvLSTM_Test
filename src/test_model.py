import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data_loader import data_tensor
from models import ConvLSTMOneStep, ConvLSTMEncoderDecoder
from config import DEVICE, DATA_PATH, ONESTEP_MODEL_PATH, MULTISTEP_MODEL_PATH
import argparse
import os


def load_models():
    """Загрузка обученных моделей"""
    onestep_model = ConvLSTMOneStep().to(DEVICE)
    if os.path.exists(ONESTEP_MODEL_PATH):
        onestep_model.load_state_dict(torch.load(ONESTEP_MODEL_PATH, map_location=DEVICE))
        print(f"Одношаговая модель загружена из {ONESTEP_MODEL_PATH}")
    else:
        print(f"Внимание: файл {ONESTEP_MODEL_PATH} не найден!")
        onestep_model = None

    multistep_model = ConvLSTMEncoderDecoder().to(DEVICE)
    if os.path.exists(MULTISTEP_MODEL_PATH):
        multistep_model.load_state_dict(torch.load(MULTISTEP_MODEL_PATH, map_location=DEVICE))
        print(f"Многошаговая модель загружена из {MULTISTEP_MODEL_PATH}")
    else:
        print(f"Внимание: файл {MULTISTEP_MODEL_PATH} не найден!")
        multistep_model = None

    return onestep_model, multistep_model


def test_onestep_model(model, data_tensor, seq_idx=None, num_frames=10):
    """Тестирование одношаговой модели"""
    if model is None:
        print("Одношаговая модель не загружена!")
        return

    model.eval()

    if seq_idx is None:
        seq_idx = torch.randint(0, data_tensor.shape[1], (1,)).item()

    print(f"Тестирование одношаговой модели на последовательности {seq_idx}")

    input_seq = data_tensor[:num_frames, seq_idx:seq_idx + 1, :, :]

    with torch.no_grad():
        predictions = model(input_seq)

        input_np = input_seq[:, 0].cpu().numpy()
        pred_np = predictions[:, 0].cpu().numpy()

        target_np = data_tensor[1:num_frames + 1, seq_idx, :, :].cpu().numpy()

    fig, axes = plt.subplots(3, num_frames, figsize=(15, 6))
    for i in range(num_frames):
        if i < len(input_np):
            axes[0, i].imshow(input_np[i], cmap='gray')
        axes[0, i].set_title(f"Вход {i + 1}")
        axes[0, i].axis('off')

        if i < len(target_np):
            axes[1, i].imshow(target_np[i], cmap='gray')
        axes[1, i].set_title(f"Цель {i + 1}")
        axes[1, i].axis('off')

        if i < len(pred_np):
            axes[2, i].imshow(pred_np[i], cmap='gray')
        axes[2, i].set_title(f"Предсказание {i + 1}")
        axes[2, i].axis('off')

    plt.suptitle(f"Одношаговая модель - Предсказание для последовательности {seq_idx}")
    plt.tight_layout()
    plt.show()

    return input_np, target_np, pred_np


def test_multistep_model(model, data_tensor, seq_idx=None, input_len=10, pred_len=10, save_gif=True):
    """Тестирование многошаговой модели с сохранением GIF"""
    if model is None:
        print("Многошаговая модель не загружена!")
        return

    model.eval()

    if seq_idx is None:
        seq_idx = torch.randint(0, data_tensor.shape[1], (1,)).item()

    print(f"Тестирование многошаговой модели на последовательности {seq_idx}")
    print(f"Вход: {input_len} кадров, Предсказание: {pred_len} кадров")

    input_seq = data_tensor[:input_len, seq_idx:seq_idx + 1, :, :]
    target_seq = data_tensor[input_len:input_len + pred_len, seq_idx:seq_idx + 1, :, :]

    with torch.no_grad():
        predictions = model(input_seq, num_prediction_steps=pred_len)

        input_np = input_seq[:, 0].cpu().numpy()
        target_np = target_seq[:, 0].cpu().numpy()
        pred_np = predictions[:, 0].cpu().numpy()

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i in range(5):
        if i < len(input_np):
            axes[0, i].imshow(input_np[i], cmap='gray')
        axes[0, i].set_title(f"Вход {i + 1}")
        axes[0, i].axis('off')

        if i < len(target_np):
            axes[1, i].imshow(target_np[i], cmap='gray')
        axes[1, i].set_title(f"Цель {i + 1}")
        axes[1, i].axis('off')

        if i < len(pred_np):
            axes[2, i].imshow(pred_np[i], cmap='gray')
        axes[2, i].set_title(f"Предсказание {i + 1}")
        axes[2, i].axis('off')

    plt.suptitle(f"Многошаговая модель - Предсказание для последовательности {seq_idx}")
    plt.tight_layout()
    plt.show()

    if save_gif:
        create_prediction_gif(input_np, target_np, pred_np, seq_idx)

    return input_np, target_np, pred_np


def create_prediction_gif(input_frames, target_frames, pred_frames, seq_idx, fps=2):
    """Создание GIF анимации предсказаний"""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    im1 = axes[0].imshow(input_frames[0], cmap='gray', vmin=0, vmax=1)
    im2 = axes[1].imshow(target_frames[0], cmap='gray', vmin=0, vmax=1)
    im3 = axes[2].imshow(pred_frames[0], cmap='gray', vmin=0, vmax=1)

    axes[0].set_title("Входные кадры")
    axes[1].set_title("Целевые кадры")
    axes[2].set_title("Предсказанные кадры")

    for ax in axes:
        ax.axis('off')

    def update(frame):
        if frame < len(input_frames):
            im1.set_array(input_frames[frame])
        else:
            im1.set_array(input_frames[-1])

        if frame < len(target_frames):
            im2.set_array(target_frames[frame])

        if frame < len(pred_frames):
            im3.set_array(pred_frames[frame])

        return im1, im2, im3

    total_frames = max(len(input_frames), len(target_frames), len(pred_frames))
    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=1000 / fps, blit=True)

    gif_filename = f"multistep_prediction_seq{seq_idx}.gif"
    ani.save(gif_filename, writer='pillow', fps=fps)
    print(f"GIF анимация сохранена как {gif_filename}")

    plt.close(fig)
    return gif_filename


def main():
    """Основная функция тестирования"""
    parser = argparse.ArgumentParser(description='Тестирование ConvLSTM моделей для MovingMNIST')
    parser.add_argument('--model', type=str, choices=['onestep', 'multistep', 'both'],
                        default='both', help='Модель для тестирования')
    parser.add_argument('--seq_idx', type=int, default=None,
                        help='Индекс тестовой последовательности (по умолчанию случайный)')
    parser.add_argument('--input_len', type=int, default=10,
                        help='Длина входной последовательности для многошаговой модели')
    parser.add_argument('--pred_len', type=int, default=10,
                        help='Длина предсказываемой последовательности для многошаговой модели')
    parser.add_argument('--no_gif', action='store_true',
                        help='Не сохранять GIF для многошаговой модели')

    args = parser.parse_args()

    onestep_model, multistep_model = load_models()

    print(f"Загрузка датасета из {DATA_PATH}")
    print(f"Размер датасета: {data_tensor.shape}")
    print(f"Всего последовательностей: {data_tensor.shape[1]}")

    if args.model in ['onestep', 'both'] and onestep_model is not None:
        print("\n" + "=" * 60)
        print("ТЕСТИРОВАНИЕ ОДНОШАГОВОЙ МОДЕЛИ")
        print("=" * 60)
        test_onestep_model(onestep_model, data_tensor, args.seq_idx)

    if args.model in ['multistep', 'both'] and multistep_model is not None:
        print("\n" + "=" * 60)
        print("ТЕСТИРОВАНИЕ МНОГОШАГОВОЙ МОДЕЛИ")
        print("=" * 60)
        test_multistep_model(multistep_model, data_tensor, args.seq_idx,
                             args.input_len, args.pred_len, not args.no_gif)


if __name__ == "__main__":
    main()