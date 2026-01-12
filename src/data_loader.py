import numpy as np
import torch
import os
import urllib.request
from config import DEVICE, DATA_PATH

print(f"Используемое устройство: {DEVICE}")

if not os.path.exists(DATA_PATH):
    print("Скачивание датасета...")
    urllib.request.urlretrieve(
        "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
        DATA_PATH
    )
    print("Датасет скачан!")

data = np.load(DATA_PATH)
print(f"Размер датасета: {data.shape}")
print(f"Размер кадра: {data.shape[2]}x{data.shape[3]}")

data = data.astype(np.float32) / 255.0
data_tensor = torch.FloatTensor(data)
if DEVICE.type == 'cuda':
    data_tensor = data_tensor.to(DEVICE)


def split_data(data_tensor, train_ratio=0.8, random_seed=42):
    """Разделение данных на train и test"""
    torch.manual_seed(random_seed)

    n_samples = data_tensor.shape[1]
    indices = torch.randperm(n_samples)
    split_idx = int(n_samples * train_ratio)

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_data = data_tensor[:, train_indices, :, :]
    test_data = data_tensor[:, test_indices, :, :]

    return train_data, test_data, train_indices, test_indices