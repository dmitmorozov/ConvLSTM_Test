import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from data_loader import data_tensor, split_data
from models import ConvLSTMEncoderDecoder
from visualization import plot_losses, visualize_predictions_comparison
from config import DEVICE, MULTISTEP_CONFIG, ONESTEP_MODEL_PATH


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
                                       plot_every=10, train_ratio=0.8):
    """Обучение многошаговой модели Encoder-Decoder"""

    print("=" * 70)
    print("ОБУЧЕНИЕ МНОГОШАГОВОЙ МОДЕЛИ ENCODER-DECODER")
    print("=" * 70)

    if model is None:
        model = ConvLSTMEncoderDecoder().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_data, test_data, _, _ = split_data(data_tensor, train_ratio)

    train_losses = []
    epoch_train_losses = []
    epoch_test_losses = []

    test_input = data_tensor[:input_seq_len, :1, :, :]
    test_target = data_tensor[input_seq_len:input_seq_len + pred_seq_len, :1, :, :]

    print(f"\nНачинаем обучение...")
    print(f"Входная последовательность: {input_seq_len} кадров")
    print(f"Предсказание: {pred_seq_len} кадров")
    print(f"Размер батча: {batch_size}")
    print(f"Learning rate: {learning_rate}")

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        epoch_train_loss = 0

        n_iterations = min(500, train_data.shape[1] // batch_size)
        pbar = tqdm(range(n_iterations), desc=f"Эпоха {epoch + 1}/{epochs}")

        for i in pbar:
            batch_idx = torch.randint(0, train_data.shape[1], (batch_size,))
            input_batch = train_data[:input_seq_len, batch_idx, :, :]
            target_batch = train_data[input_seq_len:input_seq_len + pred_seq_len, batch_idx, :, :]

            optimizer.zero_grad()

            predictions = model(input_batch, num_prediction_steps=pred_seq_len)

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
        epoch_test_loss = 0
        test_iterations = min(100, test_data.shape[1] // batch_size)
        with torch.no_grad():
            for i in range(test_iterations):
                batch_idx = torch.randint(0, test_data.shape[1], (batch_size,))

                input_batch = test_data[:input_seq_len, batch_idx, :, :]
                target_batch = test_data[input_seq_len:input_seq_len + pred_seq_len, batch_idx, :, :]

                predictions = model(input_batch, num_prediction_steps=pred_seq_len)
                loss = criterion(predictions, target_batch)
                epoch_test_loss += loss.item()



        avg_epoch_train_loss = epoch_train_loss / n_iterations
        epoch_train_losses.append(avg_epoch_train_loss)
        avg_epoch_test_loss = epoch_test_loss / test_iterations
        epoch_test_losses.append(avg_epoch_test_loss)
        epoch_time = time.time() - epoch_start

        print(f"Эпоха {epoch + 1} завершена:")
        print(f"Train loss: {avg_epoch_train_loss:.6f}")
        print(f"Test loss:  {avg_epoch_test_loss:.6f}")
        print(f"Время: {epoch_time:.1f} сек")

        if (epoch + 1) % visualize_every == 0:
            print(f"\nВизуализация результатов эпохи {epoch + 1}...")
            _ = visualize_predictions_comparison(
                model, test_input, test_target,
                model_name="Многошаговая модель",
                epoch=epoch + 1,
                num_prediction_steps=pred_seq_len
            )
            model.train()

        if (epoch + 1) % plot_every == 0:
            print(f"\nОбновление графика потерь...")
            plot_losses(epoch_train_losses, epoch_test_losses,
                        f"Многошаговая модель - Потери (эпоха {epoch + 1})")
        print("-" * 50)

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
        batch_size=MULTISTEP_CONFIG['batch_size'],
        learning_rate=MULTISTEP_CONFIG['learning_rate'],
        input_seq_len = MULTISTEP_CONFIG['input_seq_len'],
        pred_seq_len = MULTISTEP_CONFIG['pred_seq_len'],
        visualize_every=MULTISTEP_CONFIG['visualize_every'],
        plot_every=MULTISTEP_CONFIG['plot_every'],
        train_ratio=MULTISTEP_CONFIG['train_ratio']
    )
    torch.save(multistep_model.state_dict(), 'multistep_model_final.pth')