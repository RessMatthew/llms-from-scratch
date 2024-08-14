<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



## About The Project

从零开始构建大型语言模型 - 构建类GPT模型

### 模型图

![GPT模型.png](https://ressmatthew-picture-cloud-storage.oss-cn-hangzhou.aliyuncs.com/img/202408140031640.png)



### 数据流图

![GPT-2的数据流图.png](https://ressmatthew-picture-cloud-storage.oss-cn-hangzhou.aliyuncs.com/img/202408140044719.png)



## Getting Started

### Prerequisite

需安装 [conda](https://github.com/conda-forge/miniforge)，推荐安装开源Miniforge。



### Installation

```sh
# 创建名为 LLMs 的环境
conda create -n LLMs python=3.10
# 激活 LLMs 的环境
conda activate LLMs
# 依据 requirements.txt 安装依赖包
pip install -r requirements.txt
```



## Project Structure

- input：所有输入文件和数据
- src：.py脚本
- models：训练过的模型
- notebook：.ipynb文件（探索数据、绘制图表和图形）
- readme.md



## Reference

* 《build a large language model》
* [datawhalechina/llms-from-scratch-cn](https://github.com/datawhalechina/llms-from-scratch-cn)
* [rasbt/LLMs-from-scratc](https://github.com/rasbt/LLMs-from-scratch)
