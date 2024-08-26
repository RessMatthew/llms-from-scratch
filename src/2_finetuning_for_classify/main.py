
import tiktoken
from torch.utils.data import DataLoader
from transformers import GPT2Model
import torch
import time

from SpamDataset import SpamDataset
from config import model_names, BASE_CONFIG, model_configs
from LoadWeights import load_weights
from GPTModel import GPTModel
from WorkingWithText import generate, text_to_token_ids, token_ids_to_text
from Evaluate import calc_accuracy_loader, calc_loss_batch, evaluate_model

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
           optimizer.zero_grad()
           loss = calc_loss_batch(input_batch, target_batch, model, device) 
           loss.backward()
           optimizer.step() # 更新模型参数
           examples_seen += input_batch.shape[0]
           global_step += 1
           
           if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        
    return train_losses, val_losses, train_accs, val_accs, examples_seen

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 数据集
    train_dataset = SpamDataset(
        csv_file="train.csv",
        max_length=None,
        tokenizer=tokenizer
    )
    val_dataset = SpamDataset(
        csv_file="validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file="test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    
    # 数据加载
    num_workers = 0
    batch_size = 8
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    
    # 载入预训练参数
    CHOOSE_MODEL = "gpt2-small (124M)"
    gpt_hf = GPT2Model.from_pretrained(model_names[CHOOSE_MODEL], cache_dir="checkpoints")
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    model = GPTModel(BASE_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_weights(model, gpt_hf)
    
    # 微调输出层，让所有层都不可训练
    for param in model.parameters():
        param.requires_grad = False
    
    # 训练最后三层
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True
        
    model.to(device)
    

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")