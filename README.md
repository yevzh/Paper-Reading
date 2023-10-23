# Research-paper-notes

zwm的论文阅读笔记，聚焦于recommender systems, educational data mining, reinforcement learning. 笔记顺序按阅读的时间顺序记录。

## Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models

这是实验室学长姐在2023年初发表的文章，也是一个best paper。这篇文章主要聚焦于利用LLM的universal open-world knowledge来辅助进行推荐，是模型无关的。框架叫KAR，从LLMs获取两类外部知识：关于用户偏好的推理知识和关于项目的事实性知识。

KAR主要分为三个阶段：

- knowledge reasoning and generation，主要是怎么由数据集产生两方面的知识
- knowledge adaptation，主要是如何对产生的知识进行encoding和alignment
- knowledge utilization，最后应用

![kar1](.\image\kar1.png)