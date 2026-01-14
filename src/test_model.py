import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data_loader import data_tensor, split_data
from models import ConvLSTMOneStep, ConvLSTMEncoderDecoder
from config import DEVICE, DATA_PATH, ONESTEP_MODEL_PATH, MULTISTEP_MODEL_PATH
import argparse
import os
import torchmetrics
import numpy as np


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


def test_onestep_model(model, test_data, seq_idx=None, num_frames=10):
    """Тестирование одношаговой модели"""
    if model is None:
        print("Одношаговая модель не загружена!")
        return

    model.eval()

    if seq_idx is None or not (0 <= seq_idx < test_data.shape[1]):
        seq_idx = torch.randint(0, test_data.shape[1], (1,)).item()

    print(f"Тестирование одношаговой модели на последовательности {seq_idx}")

    input_seq = test_data[:num_frames, seq_idx:seq_idx + 1, :, :]

    with torch.no_grad():
        predictions = model(input_seq)

        input_np = input_seq[:, 0].cpu().numpy()
        pred_np = predictions[:, 0].cpu().numpy()

        target_np = test_data[1:num_frames + 1, seq_idx, :, :].cpu().numpy()

    print("\nРасчет метрик для одношаговой модели...")

    pred_tensor = torch.from_numpy(pred_np).unsqueeze(1).to(DEVICE)
    target_tensor = torch.from_numpy(target_np).unsqueeze(1).to(DEVICE)

    rmse_metric = torchmetrics.MeanSquaredError(squared=False).to(DEVICE)
    psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    rmse_values = []
    psnr_values = []
    ssim_values = []

    for i in range(num_frames):
        frame_pred = pred_tensor[i:i + 1]
        frame_target = target_tensor[i:i + 1]

        rmse = rmse_metric(frame_pred, frame_target).item()
        psnr = psnr_metric(frame_pred, frame_target).item()
        ssim = ssim_metric(frame_pred, frame_target).item()

        rmse_values.append(rmse)
        psnr_values.append(psnr)
        ssim_values.append(ssim)

    avg_rmse = np.mean(rmse_values)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"Средние метрики по кадрам:")
    print(f"  RMSE: {avg_rmse:.4f}")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")

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

    suptitle = f"Одношаговая модель - Последовательность {seq_idx}\n"
    suptitle += f"Средние метрики: RMSE={avg_rmse:.4f}, PSNR={avg_psnr:.2f} дБ, SSIM={avg_ssim:.4f}"
    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\nСоздание GIF анимации для одношаговой модели...")
    create_onestep_prediction_gif(input_np, target_np, pred_np, seq_idx)

    return input_np, target_np, pred_np


def test_multistep_model(model, test_data, seq_idx=None, input_len=10, pred_len=10, save_gif=True):
    """Тестирование многошаговой модели с сохранением GIF"""
    if model is None:
        print("Многошаговая модель не загружена!")
        return

    model.eval()

    if seq_idx is None:
        seq_idx = torch.randint(0, test_data.shape[1], (1,)).item()

    print(f"Тестирование многошаговой модели на последовательности {seq_idx}")
    print(f"Вход: {input_len} кадров, Предсказание: {pred_len} кадров")

    input_seq = test_data[:input_len, seq_idx:seq_idx + 1, :, :]
    target_seq = test_data[input_len:input_len + pred_len, seq_idx:seq_idx + 1, :, :]

    with torch.no_grad():
        predictions = model(input_seq, num_prediction_steps=pred_len)

        input_np = input_seq[:, 0].cpu().numpy()
        target_np = target_seq[:, 0].cpu().numpy()
        pred_np = predictions[:, 0].cpu().numpy()

    fig, axes = plt.subplots(3, 10, figsize=(15, 9))
    for i in range(10):
        if i < len(input_np):
            axes[0, i].imshow(input_np[i], cmap='gray')
            axes[0, i].set_title(f"Вход {i + 1}")
        else:
            axes[0, i].imshow(input_np[-1], cmap='gray')
            axes[0, i].set_title(f"Вход {i + 1} (последний)")
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

    input_lengths_range = range(1, 11)
    prediction_steps = 10

    rmse_matrix = np.zeros((len(input_lengths_range), prediction_steps))
    psnr_matrix = np.zeros((len(input_lengths_range), prediction_steps))
    ssim_matrix = np.zeros((len(input_lengths_range), prediction_steps))

    print("Расчет метрик для разных длин входной последовательности...")

    for input_len_idx, current_input_len in enumerate(input_lengths_range):

        current_input_seq = test_data[:current_input_len, seq_idx:seq_idx + 1, :, :]
        current_target_seq = test_data[current_input_len:current_input_len + prediction_steps,
                             seq_idx:seq_idx + 1, :, :]

        with torch.no_grad():
            current_predictions = model(current_input_seq, num_prediction_steps=prediction_steps)

            current_pred_tensor = current_predictions
            current_target_tensor = current_target_seq.to(DEVICE)

        for step in range(prediction_steps):
            frame_pred = current_pred_tensor[step:step + 1]
            frame_target = current_target_tensor[step:step + 1]

            rmse_metric = torchmetrics.MeanSquaredError(squared=False).to(DEVICE)
            psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
            ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

            rmse = rmse_metric(frame_pred, frame_target).item()
            psnr = psnr_metric(frame_pred, frame_target).item()
            ssim = ssim_metric(frame_pred, frame_target).item()

            rmse_matrix[input_len_idx, step] = rmse
            psnr_matrix[input_len_idx, step] = psnr
            ssim_matrix[input_len_idx, step] = ssim

    avg_rmse_by_step = rmse_matrix.mean(axis=0)
    avg_psnr_by_step = psnr_matrix.mean(axis=0)
    avg_ssim_by_step = ssim_matrix.mean(axis=0)

    prediction_steps_range = range(1, prediction_steps + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(prediction_steps_range, avg_rmse_by_step, 'bo-', linewidth=2, markersize=6)
    axes[0].set_xlabel('Номер предсказанного кадра')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Усредненный RMSE по входным длинам')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(prediction_steps_range)

    axes[1].plot(prediction_steps_range, avg_psnr_by_step, 'go-', linewidth=2, markersize=6)
    axes[1].set_xlabel('Номер предсказанного кадра')
    axes[1].set_ylabel('PSNR (дБ)')
    axes[1].set_title('Усредненный PSNR по входным длинам')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(prediction_steps_range)

    axes[2].plot(prediction_steps_range, avg_ssim_by_step, 'mo-', linewidth=2, markersize=6)
    axes[2].set_xlabel('Номер предсказанного кадра')
    axes[2].set_ylabel('SSIM')
    axes[2].set_title('Усредненный SSIM по входным длинам')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(prediction_steps_range)

    plt.suptitle(f"Многошаговая модель - Зависимость метрик от номера предсказанного кадра\n"
                 f"(усреднено по входным длинам 1-10, последовательность {seq_idx})",
                 fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\nСредние метрики по всем предсказанным кадрам (усреднено по входным длинам):")
    print(f"  RMSE: {avg_rmse_by_step.mean():.4f}")
    print(f"  PSNR: {avg_psnr_by_step.mean():.2f} dB")
    print(f"  SSIM: {avg_ssim_by_step.mean():.4f}")

    if save_gif:
        create_prediction_gif_multistep(input_np, target_np, pred_np, seq_idx)

    return input_np, target_np, pred_np


def create_prediction_gif_multistep(input_frames, target_frames, pred_frames, seq_idx, fps=2):
    """Создание GIF анимации предсказаний для многошаговой модели"""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    im1 = axes[0].imshow(input_frames[0], cmap='gray', vmin=0, vmax=1)
    im2 = axes[1].imshow(target_frames[0], cmap='gray', vmin=0, vmax=1)
    im3 = axes[2].imshow(pred_frames[0], cmap='gray', vmin=0, vmax=1)

    axes[0].set_title("Входные кадры (0-9)")
    axes[1].set_title("Целевые кадры (10-19)")
    axes[2].set_title("Предсказанные кадры (10-19)")

    for ax in axes:
        ax.axis('off')

    def update(frame):
        total_input_frames = len(input_frames)
        total_pred_frames = len(pred_frames)

        if frame < total_input_frames:
            im1.set_array(input_frames[frame])
        elif frame - total_input_frames < total_pred_frames:
            im1.set_array(pred_frames[frame - total_input_frames])

        if frame < total_pred_frames:
            im2.set_array(target_frames[frame])

        if frame < total_pred_frames:
            im3.set_array(pred_frames[frame])

        return im1, im2, im3

    total_frames = len(input_frames) + len(pred_frames)
    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=1000 / fps, blit=True)

    gif_filename = f"multistep_prediction_seq{seq_idx}.gif"
    ani.save(gif_filename, writer='pillow', fps=fps)
    print(f"GIF анимация сохранена как {gif_filename}")

    plt.close(fig)
    return gif_filename


def create_onestep_prediction_gif(input_frames, target_frames, pred_frames, seq_idx, fps=2):
    """Создание GIF анимации предсказаний для одношаговой модели"""
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
            im1.set_array(pred_frames[frame - len(input_frames)])

        if frame < len(target_frames):
            im2.set_array(target_frames[frame])

        if frame < len(pred_frames):
            im3.set_array(pred_frames[frame])

        return im1, im2, im3

    total_frames = len(pred_frames)
    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=1000 / fps, blit=True)

    gif_filename = f"onestep_prediction_seq{seq_idx}.gif"
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

    train_data, val_data, test_data, _, _, _ = split_data(data_tensor, ratios=(0.7, 0.15, 0.15))

    print(f"Загрузка датасета из {DATA_PATH}")
    print(f"Размер тестовых данных: {test_data.shape}")
    print(f"Всего тестовых последовательностей: {test_data.shape[1]}")

    if args.model in ['onestep', 'both'] and onestep_model is not None:
        print("\n" + "=" * 60)
        print("ТЕСТИРОВАНИЕ ОДНОШАГОВОЙ МОДЕЛИ")
        print("=" * 60)
        test_onestep_model(onestep_model, test_data, args.seq_idx)

    if args.model in ['multistep', 'both'] and multistep_model is not None:
        print("\n" + "=" * 60)
        print("ТЕСТИРОВАНИЕ МНОГОШАГОВОЙ МОДЕЛИ")
        print("=" * 60)
        test_multistep_model(multistep_model, test_data, args.seq_idx,
                             args.input_len, args.pred_len, not args.no_gif)


if __name__ == "__main__":
    main()