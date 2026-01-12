import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = "data/mnist_test_seq.npy"
ONESTEP_MODEL_PATH = "models/onestep/onestep_model.pth"
MULTISTEP_MODEL_PATH = "models/multistep/multistep_model.pth"

ONESTEP_CONFIG = {
    'epochs': 50,
    'batch_size': 16,
    'learning_rate': 0.001,
    'visualize_every': 10,
    'plot_every': 10,
    'train_ratio': 0.8
}

MULTISTEP_CONFIG = {
    'epochs': 50,
    'batch_size': 16,
    'learning_rate': 0.001,
    'input_seq_len': 10,
    'pred_seq_len': 10,
    'visualize_every': 10,
    'plot_every': 10,
    'train_ratio': 0.8
}