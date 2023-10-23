# Research-paper-notes

zwm的论文阅读笔记，聚焦于recommender systems, educational data mining, reinforcement learning. 笔记顺序按阅读的时间顺序记录。

## Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models

这是实验室学长姐在2023年初发表的文章，也是一个best paper。这篇文章主要聚焦于利用LLM的universal open-world knowledge来辅助进行推荐，是模型无关的。框架叫KAR，从LLMs获取两类外部知识：关于用户偏好的推理知识和关于项目的事实性知识。

KAR主要分为三个阶段：

- knowledge reasoning and generation，主要是怎么由数据集产生两方面的知识
- knowledge adaptation，主要是如何对产生的知识进行encoding和alignment
- knowledge utilization，将augmented vectors与现有推荐模型结合

![kar1](./image/kar1.png)

这篇文章我主要要学习的是adaptation模块，这边分为两个部分：knowledge encoder和hybrid-expert adaptor。knowledge encoder很好理解，我也实现了用大语言模型抽取隐层。后者使用了一个MoE模块，聚合不同方面的知识(**what does the "experts" represent here**)

![kar1](./image/kar2.png)

对于如何应用进CTR模型，直接将这些增广向量作为额外的输入特征，允许它们显式地与其他类别特征进行交互。

我觉得这篇文章的壁垒主要在于太贵了，如果生成文本的阶段都用gpt去生成那费用恐怖如斯，但是又不知道一般的13B大模型能不能有与之媲美的生成效果。针对用户情况和题目情况分别生成文本时可借鉴的，两段文本的表征进行聚合后如何嵌入CDM模型也是需要思考的。

## Towards Universal Sequence Representation Learning for Recommender Systems

这是人大和阿里发表的一篇文章，提出了还算有名的UniSRec，也是利用项目的关联描述文本来学习跨不同场景的可迁移表示。其实我不太懂这里cross-domain和transfer learning的东西，只是想看一下这篇文章是如何将文本利用起来的。

这篇文章主要通过两方面的学习：

- 通用商品表征：通过对商品的文本进行BERT编码得到对应的表征；为了得到同性语义并且环节不同域之间的语义差异，作者提出带参数的白话特征进行变换，用MoE的方法进行处理
- 通用序列表征：通过Transformer对序列进行编码，为了缓解不同域中的行为序列的反应的行为模式的差异性，以及缓解跷跷板问题。对比学习构建序列-商品对比任务和序列序列对比任务捕获相关性。

![kar1](./image/unisrec1.png)

一些没见过的点：

- 参数白化进行语义转换：BERT为一般文本引入了非平滑的各向异性语义空间。于是用线性变换变换原始BERT表征。

- 通过MoE进行域融合和适应：为一个商品学习多个白化embedding，利用这些embedding的自适应组合作为通用商品表征。

- sequence-item对比学习：对于给定的序列，采用跨域商品作为负样本。这里s是指序列，v是指应该的下一个商品。B的batch是来自不同域的。

  ![kar1](./image/unisrec2.png)

- 相比较而言，sequence-sequence用的就是老套路了，增强序列与其他序列进行对比：

  ![kar1](./image/unisrec3.png)

