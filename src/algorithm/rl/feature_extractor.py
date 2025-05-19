import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, Tuple, Any, List
from torch import Tensor

from src.algorithm.rl.agent import PathFindingAgent

class NormalExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict[str, Tensor], features_dim: int = 128) -> None: 
        features_dim = observation_space['SensorObservation'].shape[0]
        features_dim += observation_space['ActionObservation'].shape[0]
        features_dim += observation_space['PanelConfigObservation'].shape[0]
        super(NormalExtractor, self).__init__(observation_space, features_dim)

    
    def forward(self, observations: Dict[str, Tensor]) -> Tensor:
        sensor = observations['SensorObservation']
        action = observations['ActionObservation']
        panel_config = observations['PanelConfigObservation']
        
        
        # Concatenate features
        final_features = torch.cat((sensor, action, panel_config), dim=1)    
        # Pass through final linear layer  
        return final_features

class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)))

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return x

class PathFindingFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict[str, Tensor], features_dim: int = 256) -> None: 
        super(PathFindingFeatureExtractor, self).__init__(observation_space, features_dim)
        self.panel_config_mlp = nn.Sequential(
            nn.Linear(45, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU())
        self.sensor_mlp = nn.Sequential(
            nn.Linear(56, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU())
        self.action_mlp = nn.Sequential(    
                nn.Linear(6, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 32),
                nn.LeakyReLU())
        
        self.cnn = CNN3D()  # Initialize the 3D CNN model 
        self.final_linear = nn.Linear(128 + 128, features_dim)
    
    
    def forward(self, observations: Dict[str, Tensor]) -> Tensor:
        voxel = observations['VoxelObservation']    
        sensor = observations['SensorObservation']
        action = observations['ActionObservation']
        panel_config = observations['PanelConfigObservation']
        
        voxel = voxel.reshape(-1, 3, 30, 30, 30)
        voxel_features = self.cnn(voxel)
        
        # Extract features from each observation
        sensor_features = self.sensor_mlp(sensor)
        action_features = self.action_mlp(action)
        panel_config_features = self.panel_config_mlp(panel_config)
        
        # Concatenate features
        final_features = torch.cat((voxel_features, sensor_features, action_features, panel_config_features), dim=1)    
        # Pass through final linear layer
        return self.final_linear(final_features)

class Slice2DCNN(nn.Module):
    """z축 방향 2D 슬라이스를 위한 공유 2D CNN"""
    def __init__(self, in_channels=3, feature_dim=128):
        super(Slice2DCNN, self).__init__()
        self.cnn2d = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # output: (batch, 64, 1, 1)
        )
        self.fc = nn.Linear(64, feature_dim)

    def forward(self, x2d):
        # x2d: (batch * num_slices, C, H, W)
        out = self.cnn2d(x2d)
        out = out.view(out.size(0), -1)  # (batch * num_slices, 64)
        out = self.fc(out)              # (batch * num_slices, feature_dim)
        return out

class AttentionAggregator(nn.Module):
    def __init__(self, feature_dim=128, num_heads=4):
        super(AttentionAggregator, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
    
    def forward(self, features):
        # features: (batch, num_slices, feature_dim)
        attn_output, _ = self.attention(features, features, features)  # self-attention
        pooled = attn_output.mean(dim=1)  # 평균 풀링 or [CLS] 대체 가능
        return pooled

class CNN2DWithAttention3D(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128):
        super(CNN2DWithAttention3D, self).__init__()
        self.slice_encoder = Slice2DCNN(in_channels, feature_dim)
        self.attn = AttentionAggregator(feature_dim)

    def forward(self, x):
        # x: (B, C, H, W, D)
        B, C, H, W, D = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, D, C, H, W)
        x = x.view(B * D, C, H, W)  # (B*D, C, H, W)

        encoded = self.slice_encoder(x)  # (B*D, feature_dim)
        encoded = encoded.view(B, D, -1)  # (B, D, feature_dim)

        aggregated = self.attn(encoded)  # (B, feature_dim)
        return aggregated