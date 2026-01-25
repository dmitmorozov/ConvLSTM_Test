import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from data_loader import data_tensor, split_data
from models import ConvLSTMEncoderDecoder
from visualization import plot_metrics, visualize_predictions_comparison_multistep
from config import DEVICE, MULTISTEP_CONFIG, ONESTEP_MODEL_PATH
import torchmetrics


def init_multistep_from_onestep(model, onestep_model_path):
    """Инициализация энкодера многошаговой модели весами из одношаговой модели"""
    print("Инициализация энкодера весами из одношаговой модели...")

    onestep_state_dict = torch.load(onestep_model_path, map_location=DEVICE)
    multistep_state_dict = model.state_dict()

    transferred_keys = 0
    for key in onestep_state_dict:
        if key.startswith('lstm1.'):
            new_key = key.replace('lstm1.', 'encoder_lstm1.')
            if new_key in multistep_state_dict:
                multistep_state_dict[new_key] = onestep_state_dict[key]
                transferred_keys += 1

        elif key.startswith('lstm2.'):
            new_key = key.replace('lstm2.', 'encoder_lstm2.')
            if new_key in multistep_state_dict:
                multistep_state_dict[new_key] = onestep_state_dict[key]
                transferred_keys += 1

    model.load_state_dict(multistep_state_dict, strict=False)

    return model



def train_multistep(model=None, epochs=50, batch_size=16,
                                       learning_rate=0.001, input_seq_len=10,
                                       pred_seq_len=10, visualize_every=5,
                                       plot_every=10, train_ratio=(0.7, 0.15, 0.15), teacher_forcing_ratio=0.5):
    """Обучение многошаговой модели Encoder-Decoder"""

    print("=" * 70)
    print("ОБУЧЕНИЕ МНОГОШАГОВОЙ МОДЕЛИ ENCODER-DECODER")
    print("=" * 70)

    if model is None:
        model = ConvLSTMEncoderDecoder().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_data, val_data, test_data, _, _, _ = split_data(data_tensor, train_ratio)

    train_rmse = torchmetrics.MeanSquaredError(squared=False).to(DEVICE)
    train_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    train_ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    val_rmse = torchmetrics.MeanSquaredError(squared=False).to(DEVICE)
    val_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    val_ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    train_losses = []
    epoch_train_losses = []
    epoch_test_losses = []

    train_rmse_history = []
    train_psnr_history = []
    train_ssim_history = []
    val_rmse_history = []
    val_psnr_history = []
    val_ssim_history = []

    test_input = data_tensor[:input_seq_len, :1, :, :]
    test_target = data_tensor[input_seq_len:input_seq_len + pred_seq_len, :1, :, :]

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3

    print(f"\nНачинаем обучение...")
    print(f"Входная последовательность: {input_seq_len} кадров")
    print(f"Предсказание: {pred_seq_len} кадров")
    print(f"Размер батча: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    n_samples = train_data.shape[1]
    indices = torch.randperm(n_samples)

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        epoch_train_loss = 0

        train_rmse.reset()
        train_psnr.reset()
        train_ssim.reset()

        n_iterations = train_data.shape[1] // batch_size
        pbar = tqdm(range(n_iterations), desc=f"Эпоха {epoch + 1}/{epochs}")

        current_tf_ratio = max(teacher_forcing_ratio * 0.2, teacher_forcing_ratio * (1 - epoch / epochs))

        for i in pbar:
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            if start_idx >= n_samples:  # Re-shuffle for next epoch
                indices = torch.randperm(n_samples)
                start_idx = 0
                end_idx = batch_size
            batch_idx = indices[start_idx:end_idx]

            input_batch = train_data[:input_seq_len, batch_idx, :, :]
            target_batch = train_data[input_seq_len:input_seq_len + pred_seq_len, batch_idx, :, :]

            optimizer.zero_grad()

            predictions = model(input_batch, target_sequences=target_batch, num_prediction_steps=pred_seq_len,teacher_forcing_ratio=current_tf_ratio)

            loss = criterion(predictions, target_batch)

            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            epoch_train_loss += current_loss
            train_losses.append(current_loss)

            with torch.no_grad():
                pred_flat = predictions.reshape(-1, 1, predictions.shape[-2], predictions.shape[-1])
                target_flat = target_batch.reshape(-1, 1, target_batch.shape[-2], target_batch.shape[-1])

                train_rmse.update(pred_flat, target_flat)
                train_psnr.update(pred_flat, target_flat)

                if (epoch + 1) % 5 == 0:
                    train_ssim.update(pred_flat, target_flat)

            pbar.set_postfix({
                'train_loss': f'{current_loss:.4f}',
                'avg_train': f'{epoch_train_loss / (i + 1):.4f}'
            })

        train_rmse_value = train_rmse.compute().item()
        train_psnr_value = train_psnr.compute().item()

        model.eval()
        epoch_val_loss = 0
        val_iterations = val_data.shape[1] // batch_size

        val_rmse.reset()
        val_psnr.reset()
        val_ssim.reset()

        with torch.no_grad():
            for i in range(val_iterations):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, val_data.shape[1])
                batch_idx = torch.arange(start_idx, end_idx)

                input_batch = val_data[:input_seq_len, batch_idx, :, :]
                target_batch = val_data[input_seq_len:input_seq_len + pred_seq_len, batch_idx, :, :]

                predictions = model(input_batch, target_sequences=None, num_prediction_steps=pred_seq_len, teacher_forcing_ratio=0.0)
                loss = criterion(predictions, target_batch)
                epoch_val_loss += loss.item()

                pred_flat = predictions.reshape(-1, 1, predictions.shape[-2], predictions.shape[-1])
                target_flat = target_batch.reshape(-1, 1, target_batch.shape[-2], target_batch.shape[-1])

                val_rmse.update(pred_flat, target_flat)
                val_psnr.update(pred_flat, target_flat)

                if (epoch + 1) % 5 == 0:
                    val_ssim.update(pred_flat, target_flat)

        val_rmse_value = val_rmse.compute().item()
        val_psnr_value = val_psnr.compute().item()

        avg_epoch_train_loss = epoch_train_loss / n_iterations
        epoch_train_losses.append(avg_epoch_train_loss)
        avg_epoch_test_loss = epoch_val_loss / val_iterations
        epoch_test_losses.append(avg_epoch_test_loss)
        epoch_time = time.time() - epoch_start

        train_rmse_history.append(train_rmse_value)
        train_psnr_history.append(train_psnr_value)
        val_rmse_history.append(val_rmse_value)
        val_psnr_history.append(val_psnr_value)

        print(f"Эпоха {epoch + 1} завершена:")
        print(f"Train loss: {avg_epoch_train_loss:.6f}")
        print(f"Val loss:  {avg_epoch_test_loss:.6f}")
        print(f"Train RMSE: {train_rmse_value:.4f}, Train PSNR: {train_psnr_value:.2f} dB")
        print(f"Val RMSE:   {val_rmse_value:.4f}, Val PSNR:   {val_psnr_value:.2f} dB")
        if (epoch + 1) % 5 == 0:
            train_ssim_value = train_ssim.compute().item()
            train_ssim_history.append(train_ssim_value)
            val_ssim_value = val_ssim.compute().item()
            val_ssim_history.append(val_ssim_value)
            print(f"Train SSIM: {train_ssim_value:.4f}, Val SSIM:   {val_ssim_value:.4f}")
        print(f"Время: {epoch_time:.1f} сек")

        if (epoch + 1) % visualize_every == 0:
            print(f"\nВизуализация результатов эпохи {epoch + 1}...")
            _ = visualize_predictions_comparison_multistep(
                model, test_input, test_target,
                model_name="Многошаговая модель",
                epoch=epoch + 1,
                num_prediction_steps=pred_seq_len
            )
            model.train()

        if (epoch + 1) % plot_every == 0:
            print(f"\nОбновление графика потерь...")
            plot_metrics(epoch_train_losses, epoch_test_losses,
                        f"Многошаговая модель - Потери (эпоха {epoch + 1})")
            plot_metrics(train_rmse_history, val_rmse_history, title=f"Многошаговая модель - RMSE (эпоха {epoch + 1})",
                         metric="RMSE")
            plot_metrics(train_psnr_history, val_psnr_history, title=f"Многошаговая модель - PSNR (эпоха {epoch + 1})",
                         metric="PSNR")
            if len(train_ssim_history) > 0:
                plot_metrics(train_ssim_history, val_ssim_history,
                             title=f"Многошаговая модель - SSIM (эпоха {epoch + 1}, каждые 5 эпох)", metric="SSIM")
        print("-" * 50)

        if epoch + 1 >= 10:
            if avg_epoch_test_loss < best_val_loss:
                best_val_loss = avg_epoch_test_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n" + "=" * 70)
                    print(f"РАННЯЯ ОСТАНОВКА")
                    print(f"Val loss не улучшался {patience} эпох подряд")
                    print(f"Лучший val loss: {best_val_loss:.6f}")
                    print("=" * 70)
                    break

    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ МНОГОШАГОВОЙ МОДЕЛИ ЗАВЕРШЕНО!")
    print("=" * 70)

    return model, train_losses, epoch_train_losses, epoch_test_losses

if __name__ == "__main__":
    multistep_model = ConvLSTMEncoderDecoder().to(DEVICE)

    multistep_model = init_multistep_from_onestep(
        multistep_model,
        ONESTEP_MODEL_PATH
    )

    multistep_model, multistep_iter_losses, multistep_epoch_train, multistep_epoch_test = train_multistep(
        epochs=MULTISTEP_CONFIG['epochs'],
        batch_size=100,
        learning_rate=MULTISTEP_CONFIG['learning_rate'],
        input_seq_len = MULTISTEP_CONFIG['input_seq_len'],
        pred_seq_len = MULTISTEP_CONFIG['pred_seq_len'],
        visualize_every=5,
        plot_every=5,
    )
    torch.save(multistep_model.state_dict(), 'multistep_model_final.pth')