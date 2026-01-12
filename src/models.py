import torch
import torch.nn as nn
from config import DEVICE

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Входные ворота
        self.Wxi = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False)
        self.w_ci = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

        # Забывающие ворота
        self.Wxf = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False)
        self.w_cf = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

        # Выходные ворота
        self.Wxo = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False)
        self.w_co = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

        # Ячейка памяти
        self.Wxc = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False)

    def forward(self, x, prev_state):
        batch_size, _, height, width = x.size()

        if prev_state is None:
            h_prev = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
            c_prev = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
        else:
            h_prev, c_prev = prev_state

        # Входные ворота
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h_prev) + self.w_ci * c_prev)

        # Забывающие ворота
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h_prev) + self.w_cf * c_prev)

        # Ячейка памяти
        c = f * c_prev + i * torch.tanh(self.Wxc(x) + self.Whc(h_prev))

        # Выходные ворота
        o = torch.sigmoid(self.Wxo(x) + self.Who(h_prev) + self.w_co * c)

        # Скрытое состояние
        h = o * torch.tanh(c)

        return h, (h, c)

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = next(self.parameters()).device
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=device))


class ConvLSTMOneStep(nn.Module):
    """Одношаговая модель"""

    def __init__(self):
        super(ConvLSTMOneStep, self).__init__()

        self.lstm1 = ConvLSTMCell(1, 32)
        self.lstm2 = ConvLSTMCell(32, 16)

        self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 1, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, input_sequences):
        x = input_sequences.unsqueeze(2)  # [time, batch, 1, height, width]

        h1_state = None
        h2_state = None

        all_predictions = []
        time_steps = x.size(0)

        for t in range(time_steps):
            frame = x[t]

            h1, h1_state = self.lstm1(frame, h1_state)
            h2, h2_state = self.lstm2(h1, h2_state)

            # Output layers
            out = self.relu(self.conv1(h2))
            out = self.relu(self.conv2(out))
            out = self.conv3(out)

            all_predictions.append(out.squeeze(1))

        return torch.stack(all_predictions, dim=0)


class ConvLSTMEncoderDecoder(nn.Module):
    """Многошаговая модель Encoder-Decoder"""

    def __init__(self):
        super(ConvLSTMEncoderDecoder, self).__init__()

        self.encoder_lstm1 = ConvLSTMCell(1, 32)
        self.encoder_lstm2 = ConvLSTMCell(32, 16)

        self.decoder_lstm1 = ConvLSTMCell(1, 16)
        self.decoder_lstm2 = ConvLSTMCell(16, 32)

        # Output layers
        self.conv1 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 1, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, input_sequences, num_prediction_steps=10):
        x = input_sequences.unsqueeze(2)  # [input_seq_len, batch, 1, height, width]

        encoder_state1 = None
        encoder_state2 = None

        for t in range(x.size(0)):
            frame = x[t]

            h1, encoder_state1 = self.encoder_lstm1(frame, encoder_state1)
            h2, encoder_state2 = self.encoder_lstm2(h1, encoder_state2)


        h_enc1, c_enc1 = encoder_state1
        h_enc2, c_enc2 = encoder_state2
        # Инициализируем состояния декодера в обратном порядке
        decoder_state1 = (h_enc2, c_enc2)
        decoder_state2 = (h_enc1, c_enc1)

        all_predictions = []
        current_input = x[-1]

        for step in range(num_prediction_steps):
            d1, decoder_state1 = self.decoder_lstm1(current_input, decoder_state1)
            d2, decoder_state2 = self.decoder_lstm2(d1, decoder_state2)

            # Output layers
            out = self.relu(self.conv1(d2))
            out = self.relu(self.conv2(out))
            out = self.conv3(out)

            prediction = out.squeeze(1)
            all_predictions.append(prediction)
            current_input = out

        return torch.stack(all_predictions, dim=0)

test_model_onestep = ConvLSTMOneStep().to(DEVICE)
test_model_multistep = ConvLSTMEncoderDecoder().to(DEVICE)