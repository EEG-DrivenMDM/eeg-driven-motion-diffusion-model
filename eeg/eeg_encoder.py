import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class EEGEncoder(nn.Module):
    def __init__(self, num_channels=8, num_heads=8, num_layers=8, time_len=512, mlp_ratio=1., norm_layer=nn.LayerNorm,
                 mha_use=False, lstm_use=True):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.time_len = time_len
        self.mha_use = mha_use
        self.lstm_use = lstm_use
        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            # nn.AvgPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
        )
        if self.mha_use:
            self.blocks = nn.ModuleList([
                Block(64, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(num_layers)])
        if self.lstm_use:
            self.lstm = nn.LSTM(input_size=64, hidden_size=64, bidirectional=False, num_layers=num_layers // 2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(time_len * 64 // (2 ** 5), 512),
        )

    def forward(self, x):
        # print(x)
        # print(x.shape)
        x = self.dropout(x)
        x = self.conv(x)
        # print(x.shape)
        if self.mha_use:
            x = x.permute(0, 2, 1)
            for blk in self.blocks:
                x = blk(x)
            x = x.permute(0, 2, 1)
        if self.lstm_use:
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
            x = x.permute(0, 2, 1)
        x = self.dropout(x)
        # print(x.shape)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    bs = 8
    channels = 20
    time_len = 512
    eeg_encoder = EEGEncoder(num_channels=channels,time_len=time_len)

    eeg = torch.randn((bs, channels, time_len))
    print(eeg.shape)
    out = eeg_encoder(eeg)
    print(out.shape)
