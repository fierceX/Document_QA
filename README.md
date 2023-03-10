# Document_QA

根据传入的文本文件，回答你的问题。

核心逻辑来自于chatPDF，自动化客服AI，以及：[ChatWeb](https://github.com/SkywalkerDarren/chatWeb)

由于原来的ChatWeb项目使用的是pqsql作为向量存储和计算工具，较为复杂，本项目修改成faiss，更简单快速。


# 基本原理

1. 读取文件，并进行分割
2. 对于每段文本，使用text-embedding-ada-002生成特征向量
3. 将向量和文本对应关系存入本地pkl文件
4. 对于用户输入，生成向量
5. 使用向量数据库进行最近邻搜索，返回最相似的文本列表
6. 使用gpt3.5的chatAPI，设计prompt，使其基于最相似的文本列表进行回答

就是先把大量文本中提取相关内容，再进行回答，最终可以达到类似突破token限制的效果  
后续可以考虑将openai的文本向量改成自定义的向量生成工具

# 准备开始

- 项目依赖

主要依赖
```
faiss
numpy
openai
```

- 环境变量

设置`OPENAI_API_KEY`为你的openai的api key

```shell
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

- 运行

```
python Document_QA.py --input_file test.md --file_embeding test.pkl
```