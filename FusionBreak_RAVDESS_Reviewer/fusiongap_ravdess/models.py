from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioNet(nn.Module):
    feature_dim: int = 128

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def features(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.encoder(audio).flatten(1)
        return x

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.features(audio)))


class VideoNet(nn.Module):
    feature_dim: int = 128

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.temporal = nn.Sequential(
            nn.Linear(128 * 2, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(160, self.feature_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def features(self, video: torch.Tensor) -> torch.Tensor:
        # video: B,T,1,H,W
        b, t, c, h, w = video.shape
        x = video.reshape(b * t, c, h, w)
        f = self.frame_encoder(x).flatten(1).reshape(b, t, -1)
        f_mean = f.mean(dim=1)
        f_max = f.max(dim=1).values
        return self.temporal(torch.cat([f_mean, f_max], dim=1))

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.features(video)))


class FusionNet(nn.Module):
    def __init__(self, audio_model: AudioNet, video_model: VideoNet, architecture: str, num_classes: int = 8):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.architecture = architecture
        d = audio_model.feature_dim

        if architecture == "concat":
            in_dim = d * 2
            self.head = self._mlp(in_dim, num_classes)
        elif architecture == "interaction":
            in_dim = d * 4
            self.head = self._mlp(in_dim, num_classes)
        elif architecture == "gated":
            self.gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
            self.head = self._mlp(d, num_classes)
        elif architecture == "group_gated":
            self.gate = nn.Sequential(nn.Linear(d * 2, d * 2), nn.Sigmoid())
            self.head = self._mlp(d * 2, num_classes)
        elif architecture == "late":
            self.logit_weight = nn.Parameter(torch.zeros(2))
            self.calibrator = nn.Linear(num_classes, num_classes)
        else:
            raise ValueError(f"Unknown fusion architecture: {architecture}")

    @staticmethod
    def _mlp(in_dim: int, num_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        af = self.audio_model.features(audio)
        vf = self.video_model.features(video)

        if self.architecture == "concat":
            return self.head(torch.cat([af, vf], dim=1))

        if self.architecture == "interaction":
            z = torch.cat([af, vf, torch.abs(af - vf), af * vf], dim=1)
            return self.head(z)

        if self.architecture == "gated":
            g = self.gate(torch.cat([af, vf], dim=1))
            z = g * af + (1.0 - g) * vf
            return self.head(z)

        if self.architecture == "group_gated":
            z = torch.cat([af, vf], dim=1)
            g = self.gate(z)
            return self.head(g * z)

        if self.architecture == "late":
            alog = self.audio_model.classifier(af)
            vlog = self.video_model.classifier(vf)
            w = torch.softmax(self.logit_weight, dim=0)
            return self.calibrator(w[0] * alog + w[1] * vlog)

        raise RuntimeError("Unreachable architecture branch")
