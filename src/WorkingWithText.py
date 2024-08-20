import torch

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # 添加批次维度
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # 消除批次维度
    return tokenizer.decode(flat.tolist())

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # 当前上下文的索引数组 idx，要生成的新token的最大数量 max_new_tokens
    for _ in range(max_new_tokens):
        # 使用最后上下文大小的索引数组
        idx_cond = idx[:, -context_size:]
        
        # 调用模型
        with torch.no_grad():
            logits = model(idx_cond)
        # 将 (batch, n_token, vocab_size) 转换为 (batch, vocab_size)
        logits = logits[:, -1, :]
        # 获取具有最高logits值的词汇表条目的索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True) # 形状为 (batch, 1)
        # 将采样的索引追加到运行序列
        idx = torch.cat((idx, idx_next), dim=1) # 形状为 (batch, n_tokens+1)
    return idx