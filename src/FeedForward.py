import torch.nn as nn

from GELU import GELU

# 用 GELU 实现一个小型神经网络模块
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 两个线性层和一个GELU激活函数
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
 
    def forward(self, x):
        return self.layers(x)