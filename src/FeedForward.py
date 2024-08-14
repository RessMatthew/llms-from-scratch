import torch.nn as nn

# FeedForward 前馈网络块
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear1 = nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
 
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x