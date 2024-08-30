[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

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
- notebook：.ipynb文件 - 笔记
  - ch02：数据处理
  - ch03：注意力机制
  - ch04：实现 GPT - 2 模型
  - ch05：预训练
  - ch06：文本分类微调
  - Ch07：指令微调
- src：
  - 1_pretraining：编码 GPT 模型，预训练
  - 2_finetuning_for_classify：文本分类微调
  - 3_instructions_finetuning：指令微调

- models：训练过的模型
- pdf：整理的笔记
- readme.md

## Reference

* 《build a large language model》
* [datawhalechina/llms-from-scratch-cn](https://github.com/datawhalechina/llms-from-scratch-cn)
* [rasbt/LLMs-from-scratc](https://github.com/rasbt/LLMs-from-scratch)

<!-- MARKDOWN LINKS & IMAGES -->

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/RessMatthew/llms-from-scratch.svg?style=for-the-badge
[contributors-url]: https://github.com/RessMatthew/llms-from-scratch/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/RessMatthew/llms-from-scratch.svg?style=for-the-badge
[forks-url]: https://github.com/RessMatthew/llms-from-scratch/network/members
[stars-shield]: https://img.shields.io/github/stars/RessMatthew/llms-from-scratch.svg?style=for-the-badge
[stars-url]: https://github.com/RessMatthew/llms-from-scratch/stargazers
[issues-shield]: https://img.shields.io/github/issues/RessMatthew/llms-from-scratch.svg?style=for-the-badge
[issues-url]: https://github.com/RessMatthew/llms-from-scratch/issues
[license-shield]: https://img.shields.io/github/license/RessMatthew/llms-from-scratch.svg?style=for-the-badge
[license-url]: https://github.com/RessMatthew/llms-from-scratch/blob/main/LICENSE
[contributors-shield]: https://img.shields.io/github/contributors/RessMatthew/llms-from-scratch.svg?style=for-the-badge
[contributors-url]: https://github.com/RessMatthew/llms-from-scratch/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/RessMatthew/llms-from-scratch.svg?style=for-the-badge
[forks-url]: https://github.com/RessMatthew/llms-from-scratch/network/members
[stars-shield]: https://img.shields.io/github/stars/RessMatthew/llms-from-scratch.svg?style=for-the-badge
[stars-url]: https://github.com/RessMatthew/llms-from-scratch/stargazers
[issues-shield]: https://img.shields.io/github/issues/RessMatthew/llms-from-scratch.svg?style=for-the-badge
[issues-url]: https://github.com/RessMatthew/llms-from-scratch/issues
[license-shield]: https://img.shields.io/github/license/RessMatthew/llms-from-scratch.svg?style=for-the-badge
[license-url]: https://github.com/RessMatthew/llms-from-scratch/blob/main/LICENSE
