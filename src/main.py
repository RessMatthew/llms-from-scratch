import torch
import tiktoken

from DummyGPTModel import DummyGPTModel
from config import GPT_CONFIG_124M

if __name__ == "__main__":
    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)