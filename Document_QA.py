
import openai
import faiss
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import os

def create_embeddings(input):
    """Create embeddings for the provided input."""
    result = []
    # limit about 1000 tokens per request
    lens = [len(text) for text in input]
    query_len = 0
    start_index = 0
    tokens = 0

    def get_embedding(input_slice):
        embedding = openai.Embedding.create(model="text-embedding-ada-002", input=input_slice)
        return [(text, data.embedding) for text, data in zip(input_slice, embedding.data)], embedding.usage.total_tokens

    for index, l in tqdm(enumerate(lens)):
        query_len += l
        if query_len > 4096:
            ebd, tk = get_embedding(input[start_index:index + 1])
            query_len = 0
            start_index = index + 1
            tokens += tk
            result.extend(ebd)

    if query_len > 0:
        ebd, tk = get_embedding(input[start_index:])
        tokens += tk
        result.extend(ebd)
    return result, tokens

def create_embedding(text):
    """Create an embedding for the provided text."""
    embedding = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return text, embedding.data[0].embedding

class QA():
    def __init__(self,data_embe) -> None:
        d = 1536
        index = faiss.IndexFlatL2(d)
        embe = np.array([emm[1] for emm in data_embe])
        data = [emm[0] for emm in data_embe]
        index.add(embe)
        self.index = index
        self.data = data
    def __call__(self, query):
        embedding = create_embedding(query)
        context = self.get_texts(embedding[1], limit)
        answer = self.completion(query,context)
        return answer,context
    def get_texts(self,embeding,limit):
        _,text_index = self.index.search(np.array([embeding]),limit)
        context = []
        for i in list(text_index[0]):
            context.extend(self.data[i:i+5])
        # context = [self.data[i] for i in list(text_index[0])]
        return context
    
    def completion(self,query, context):
        """Create a completion."""
        lens = [len(text) for text in context]

        maximum = 3000
        for index, l in enumerate(lens):
            maximum -= l
            if maximum < 0:
                context = context[:index + 1]
                print("超过最大长度，截断到前", index + 1, "个片段")
                break

        text = "\n".join(f"{index}. {text}" for index, text in enumerate(context))
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': 'system',
                'content': f'你是一个有帮助的AI文章助手，从下文中提取有用的内容进行回答，不能回答不在下文提到的内容，相关性从高到底排序：\n\n{text}'},
                {'role': 'user', 'content': query},
            ],
        )
        print("使用的tokens：", response.usage.total_tokens)
        return response.choices[0].message.content

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Document QA")
    parser.add_argument("--input_file", default="input.txt", dest="input_file", type=str,help="输入文件路径")
    parser.add_argument("--file_embeding", default="input_embed.pkl", dest="file_embeding", type=str,help="文件embeding文件路径")
    parser.add_argument("--print_context", action='store_true',help="是否打印上下文")
    

    args = parser.parse_args()

    if os.path.isfile(args.file_embeding):
        data_embe = pickle.load(open(args.file_embeding,'rb'))
    else:
        with open(args.input_file,'r',encoding='utf-8') as f:
            texts = f.readlines()
            texts = [text.strip() for text in texts if text.strip()]
            data_embe,tokens = create_embeddings(texts)
            pickle.dump(data_embe,open(args.file_embeding,'wb'))
            print("文本消耗 {} tokens".format(tokens))

    qa =QA(data_embe)

    limit = 10
    while True:
        query = input("请输入查询(help可查看指令)：")
        if query == "quit":
            break
        elif query.startswith("limit"):
            try:
                limit = int(query.split(" ")[1])
                print("已设置limit为", limit)
            except Exception as e:
                print("设置limit失败", e)
            continue
        elif query == "help":
            print("输入limit [数字]设置limit")
            print("输入quit退出")
            continue
        answer,context = qa(query)
        if args.print_context:
            print("已找到相关片段：")
            for text in context:
                print('\t', text)
            print("=====================================")
        print("回答如下\n\n")
        print(answer.strip())
        print("=====================================")

