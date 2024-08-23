import os
import urllib.request
from urllib.error import HTTPError  # 导入HTTPError
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def download_and_load_gpt2(model_size, models_dir):
    # 验证模型大小是否在允许的范围内
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # 定义路径
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 下载文件
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # 加载设置和参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

def download_file(url, destination):
    try:
        with urllib.request.urlopen(url) as response:
            # 从头信息中获取文件大小，如果没有则默认为0
            file_size = int(response.headers.get("Content-Length", 0))

            # 检查文件是否已存在且大小相同
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return

            # 定义读取文件的块大小
            block_size = 1024  # 1KB

            # 初始化进度条，总文件大小为最大值
            progress_bar_description = os.path.basename(url)  # 从URL中提取文件名
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                # 以二进制写模式打开目标文件
                with open(destination, "wb") as file:
                    # 以块的形式读取文件并写入目标位置
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))  # 更新进度条
    except HTTPError:
        s = (
            f"指定的URL ({url}) 不正确，无法建立网络连接，"
            "\n或请求的文件暂时不可用。\n请访问以下网站获取帮助: "
            "https://github.com/rasbt/LLMs-from-scratch/discussions/273")
        print(s)

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 初始化参数字典，每一层对应一个空字典
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历检查点中的每个变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量并删除单一维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名以提取相关部分
        variable_name_parts = name.split("/")[1:]  # 跳过 'model/' 前缀

        # 确定变量的目标字典
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # 递归访问或创建嵌套字典
        for key in variable_name_parts[1:-1]:
            # 确保 target_dict 是字典类型
            if isinstance(target_dict, dict):
                target_dict = target_dict.setdefault(key, {})
            else:
                raise ValueError(f"Expected a dictionary at {key}, got {type(target_dict)} instead.")

        # 将变量数组赋值给最后一个键
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
