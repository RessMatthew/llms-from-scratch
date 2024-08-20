import torch
import tiktoken

from GPTModel import GPTModel
from Evaluate import train_model_simple
from GPTDataset import create_dataloader_v1

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, # 阉割版，标准为1024
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1) #A
    num_epochs = 10
    file_path = "../input/the-verdict.txt"
    with open(file_path, 'r', encoding="utf-8") as file:
        text_data = file.read()

    total_characters = len(text_data)
    tokenizer = tiktoken.get_encoding("gpt2")
    total_tokens = len(tokenizer.encode(text_data))
    
    train_ration = 0.90
    split_idx = int(train_ration * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False
    )
    
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
    
    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)
    
    train_losses, val_losses, track_tokens_seen =  train_model_simple(
        model, train_loader, val_loader, optimizer, device, 
        num_epochs=num_epochs, eval_freq=5, eval_iter=1,
        start_context="Every effort moves you"
    )
