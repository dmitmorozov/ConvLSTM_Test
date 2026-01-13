import numpy as np
import torch
import os
import urllib.request
from config import DEVICE, DATA_PATH

print(f"Используемое устройство: {DEVICE}")

data_dir = os.path.dirname(DATA_PATH)
if not os.path.exists(data_dir):
    print(f"Создание директории {data_dir}...")
    os.makedirs(data_dir, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Скачивание датасета...")
    try:
        urllib.request.urlretrieve(
            "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
            DATA_PATH
        )
        print("Датасет скачан!")
    except Exception as e:
        print(f"Ошибка при скачивании датасета: {e}")
        print("Пожалуйста, скачайте датасет вручную:")
        print("wget http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy -P data/")
        exit(1)

data = np.load(DATA_PATH)
print(f"Размер датасета: {data.shape}")
print(f"Размер кадра: {data.shape[2]}x{data.shape[3]}")

data = data.astype(np.float32) / 255.0
data_tensor = torch.FloatTensor(data)
if DEVICE.type == 'cuda':
    data_tensor = data_tensor.to(DEVICE)


def split_data(data_tensor, ratios=(0.7, 0.1, 0.2), random_seed=42):
    """
    Разделение данных на train, validation и test
    """
    torch.manual_seed(random_seed)

    n_samples = data_tensor.shape[1]
    indices = torch.randperm(n_samples)

    train_ratio = ratios[0]
    val_ratio = ratios[1]
    assert abs(sum(ratios) - 1.0) < 1e-10, "Сумма долей должна равняться 1.0"

    train_split_idx = int(n_samples * train_ratio)
    val_split_idx = int(n_samples * (train_ratio + val_ratio))

    train_indices = indices[:train_split_idx]
    val_indices = indices[train_split_idx:val_split_idx]
    test_indices = indices[val_split_idx:]

    train_data = data_tensor[:, train_indices, :, :]
    val_data = data_tensor[:, val_indices, :, :]
    test_data = data_tensor[:, test_indices, :, :]

    return train_data, val_data, test_data, train_indices, val_indices, test_indices
