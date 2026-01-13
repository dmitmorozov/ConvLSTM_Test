import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from data_loader import data_tensor, split_data
from models import ConvLSTMOneStep
from visualization import plot_losses, visualize_predictions_comparison_one_step
from config import DEVICE, ONESTEP_CONFIG

def train_onestep(epochs=50, batch_size=8, learning_rate=0.001,
                                 visualize_every=5, plot_every=10, train_ratio=(0.7, 0.15, 0.15)):
    """Обучение одношаговой модели"""

    print("=" * 70)
    print("ОБУЧЕНИЕ ОДНОШАГОВОЙ МОДЕЛИ")
    print("=" * 70)

    model = ConvLSTMOneStep().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_data, val_data, test_data, _, _, _ = split_data(data_tensor, train_ratio)

    train_losses = []
    epoch_train_losses = []
    epoch_test_losses = []

    test_input = data_tensor[:10, 0, :, :]
    test_target = data_tensor[1:11, 0, :, :]

    print("\n  Начинаем обучение...")
    print(f"Размер батча: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    n_samples = train_data.shape[1]
    indices = torch.randperm(n_samples)

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        epoch_train_loss = 0

        n_iterations = train_data.shape[1] // batch_size
        pbar = tqdm(range(n_iterations), desc=f"Эпоха {epoch+1}/{epochs}")

        for i in pbar:
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            if start_idx >= n_samples:  # Re-shuffle for next epoch
                indices = torch.randperm(n_samples)
                start_idx = 0
                end_idx = batch_size
            batch_idx = indices[start_idx:end_idx]

            input_batch = train_data[:10, batch_idx, :, :]
            target_batch = train_data[1:11, batch_idx, :, :]

            optimizer.zero_grad()

            predictions = model(input_batch)

            loss = criterion(predictions, target_batch)

            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            epoch_train_loss += current_loss
            train_losses.append(current_loss)

            pbar.set_postfix({
                'train_loss': f'{current_loss:.4f}',
                'avg_train': f'{epoch_train_loss / (i + 1):.4f}'
            })



        model.eval()
        epoch_val_loss = 0
        val_iterations = val_data.shape[1] // batch_size

        with torch.no_grad():
            for i in range(val_iterations):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, val_data.shape[1])
                batch_idx = torch.arange(start_idx, end_idx)

                input_batch = val_data[:10, batch_idx, :, :]
                target_batch = val_data[1:11, batch_idx, :, :]

                predictions = model(input_batch)
                loss = criterion(predictions, target_batch)
                epoch_val_loss += loss.item()

        avg_epoch_train_loss = epoch_train_loss / n_iterations
        epoch_train_losses.append(avg_epoch_train_loss)
        avg_epoch_test_loss = epoch_val_loss / val_iterations
        epoch_test_losses.append(avg_epoch_test_loss)
        epoch_time = time.time() - epoch_start

        print(f"Эпоха {epoch + 1} завершена:")
        print(f"Train loss: {avg_epoch_train_loss:.6f}")
        print(f"Val loss:  {avg_epoch_test_loss:.6f}")
        print(f"Время: {epoch_time:.1f} сек")

        if (epoch + 1) % visualize_every == 0:
            print(f"\nВизуализация результатов эпохи {epoch + 1}...")
            _ = visualize_predictions_comparison_one_step(
                model, test_input, test_target,
                model_name="Одношаговая модель",
                epoch=epoch + 1
            )
            model.train()

        if (epoch + 1) % plot_every == 0:
            print(f"\nОбновление графика потерь...")
            plot_losses(epoch_train_losses, epoch_test_losses,
                        f"Одношаговая модель - Потери (эпоха {epoch + 1})")

        print("-" * 50)

    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ ОДНОШАГОВОЙ МОДЕЛИ ЗАВЕРШЕНО!")
    print("=" * 70)

    return model, train_losses, epoch_train_losses, epoch_test_losses

if __name__ == "__main__":
    onestep_model, onestep_iter_losses, onestep_epoch_train, onestep_epoch_test = train_onestep(
        epochs=ONESTEP_CONFIG['epochs'],
        batch_size=100,
        learning_rate=ONESTEP_CONFIG['learning_rate'],
        visualize_every=5,
        plot_every=5
    )
    torch.save(onestep_model.state_dict(), 'onestep_model_final.pth')
