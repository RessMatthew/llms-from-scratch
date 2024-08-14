
GPT_CONFIG_124M = {
	"vocab_size": 50257,      # 50257个单词的词汇表
    "context_length": 1024,   # 最大输入向量数
    "emb_dim": 768,           # 嵌入维度
    "n_heads": 12,            # 注意力头数
    "n_layers": 12,           # transformer数量
    "drop_rate": 0.1,         # 丢弃率10%
    "qkv_bias": False         # 偏向量
}