from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioNet(nn.Module):
    def __init__(self, num_classes: int = 6, feature_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128, feature_dim), nn.ReLU(inplace=True), nn.Dropout(0.20))
        self.head = nn.Linear(feature_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class VideoNet(nn.Module):
    def __init__(self, num_classes: int = 6, feature_dim: int = 128):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.temporal = nn.GRU(input_size=128, hidden_size=feature_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.20)
        self.head = nn.Linear(feature_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,T,1,H,W
        b, t, c, h, w = x.shape
        f = self.frame_encoder(x.reshape(b * t, c, h, w)).reshape(b, t, 128)
        out, _ = self.temporal(f)
        pooled = out.mean(dim=1)
        return self.dropout(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class FusionNet(nn.Module):
    def __init__(self, audio_model: AudioNet, video_model: VideoNet, architecture: str, num_classes: int = 6, feature_dim: int = 128):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.architecture = architecture
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        if architecture == "concat":
            in_dim = feature_dim * 2
        elif architecture == "interaction":
            in_dim = feature_dim * 4
        elif architecture == "late":
            in_dim = feature_dim * 2
        elif architecture in {"gated", "group_gated"}:
            in_dim = feature_dim * 2
            self.gate = nn.Sequential(nn.Linear(in_dim, feature_dim), nn.Sigmoid())
        else:
            raise ValueError(f"Unknown fusion architecture: {architecture}")

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )
        self.late_head = nn.Linear(num_classes * 2, num_classes)

    def features(self, xa: torch.Tensor, xv: torch.Tensor) -> torch.Tensor:
        fa = self.audio_model.features(xa)
        fv = self.video_model.features(xv)
        if self.architecture == "concat":
            return torch.cat([fa, fv], dim=1)
        if self.architecture == "interaction":
            return torch.cat([fa, fv, fa * fv, torch.abs(fa - fv)], dim=1)
        if self.architecture == "late":
            return torch.cat([fa, fv], dim=1)
        if self.architecture == "gated":
            g = self.gate(torch.cat([fa, fv], dim=1))
            return torch.cat([g * fa, (1.0 - g) * fv], dim=1)
        if self.architecture == "group_gated":
            g = self.gate(torch.cat([fa, fv], dim=1))
            g = torch.round(g * 4.0) / 4.0
            return torch.cat([g * fa, (1.0 - g) * fv], dim=1)
        raise RuntimeError("Invalid fusion architecture")

    def forward(self, xa: torch.Tensor, xv: torch.Tensor) -> torch.Tensor:
        if self.architecture == "late":
            la = self.audio_model(xa)
            lv = self.video_model(xv)
            return self.late_head(torch.cat([la, lv], dim=1))
        return self.mlp(self.features(xa, xv))


def make_models(num_classes: int, arch: str) -> tuple[AudioNet, VideoNet, FusionNet]:
    audio = AudioNet(num_classes=num_classes)
    video = VideoNet(num_classes=num_classes)
    fusion = FusionNet(audio, video, arch, num_classes=num_classes)
    return audio, video, fusion
