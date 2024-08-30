import json
import tiktoken
import torch
from functools import partial
from torch.utils.data import DataLoader
from transformers import GPT2Model
import time
import re

from InstructionDataset import custom_collate_fn, InstructionDataset, format_input
from config import model_names, BASE_CONFIG, model_configs
from LoadWeights import load_weights
from previous_chapters import GPTModel, train_model_simple, plot_losses

if __name__ == "__main__":
    
    # 1 - 读取数据集
    file_path = "instruction-data.json"
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        
    
    # 2 - 数据集划分
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion
    
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    
    # 3 - 批次化,加载数据
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )
    
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    
    # 4 - 记载预训练模型
    CHOOSE_MODEL = "gpt2-medium (355M)"
    gpt_hf = GPT2Model.from_pretrained(model_names[CHOOSE_MODEL], cache_dir="checkpoints")
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    
    model = GPTModel(BASE_CONFIG)
    load_weights(model, gpt_hf, BASE_CONFIG)
    
    
    # 5 - 训练
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    model.to(device)

    num_epochs = 2

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
    # 6 - 绘制训练loss图
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    
    # 7 - 保存模型
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")