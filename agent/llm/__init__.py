import openai
from abc import abstractmethod# 抽象方法

import torch
from transformers import AutoTokenizer, AutoModel# 自动编码器和模型
import google.generativeai as genai

# 基础LLM类
class BaseLLM:
    max_token: int = 10000
    temperature: float = 0.01# 整体的概率分布，控制生成文本的随机性，越低越保守
    top_p = 0.9 #top_p，核采样，表示模型会考虑累积概率达到 90% 的词汇选项,直接截断低概率，越高越随机
    model_name = ""

    def chat(self,
             query: str,
             history: list) -> str:
        return self.get_response(query, history)
    # 获取响应
    @abstractmethod
    def get_response(self, query, history):
        return " "
    # 设置参数
    @abstractmethod
    def set_para(self, **kwargs):
        pass


# GPT-3.5模型
class GPT3_5LLM(BaseLLM):
    temperature = 0.1,
    max_token = 1000,
    model_name = 'gpt-3.5-turbo',
    streaming = False

    def __init__(self,
                 temperature=0.1,
                 max_token=1000,
                 model_name='gpt-3.5-turbo',
                 streaming=False):
        self.set_para(temperature, max_token, model_name, streaming)

    def set_para(self,
                 temperature,
                 max_token,
                 model_name,
                 streaming):
        self.model_name = model_name
        self.temperature = temperature
        self.max_token = max_token
        self.streaming = streaming

    def send(self, massages):
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=massages,
            temperature=self.temperature,
            max_tokens=self.max_token
        )
        return response.choices[0].message.content

    def send_stream(self, massages):
        client = openai.OpenAI()
        stream = client.chat.completions.create(
            model=self.model_name,
            messages=massages,
            temperature=self.temperature,
            max_tokens=self.max_token,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    @staticmethod
    def create_massages(query, history):
        # 使用system提示词
        massages = [{'role': 'system', 'content': history[0][0]}]
        # 如果有自述，则加入messages
        if history[0][1] != '':
            massages.append({'role': 'assistant', 'content': history[0][1]})
        for i in range(1, len(history)):
            # 有消息才将信息加入
            if history[i][0] != '':
                massages.append({'role': 'user', 'content': history[i][0]})
            if history[i][1] != '':
                massages.append({'role': 'assistant', 'content': history[i][1]})
        massages.append({'role': 'user', 'content': query})

        return massages

    def get_response(self, query, history):
        massages = self.create_massages(query, history)
        if self.streaming:
            return self.send_stream(massages)
        else:
            return self.send(massages)

# ChatGLM模型
class ChatGLMLLM(BaseLLM):
    tokenizer: object = None
    model: object = None

    model_name = 'chatglm-6b-int4'
    device = 'cuda'
    temperature = 0.1,
    max_token = 1000,
    streaming = False

    def __init__(self, temperature=0.1,
                 max_token=1000,
                 model_name='chatglm-6b-int4',
                 streaming=False):
        self.set_para(temperature, max_token, model_name, streaming)

    def set_para(self,
                 temperature,
                 max_token,
                 model_name,
                 streaming):
        self.model_name = model_name
        self.temperature = temperature
        self.max_token = max_token
        self.streaming = streaming

    def set_device(self, device):
        self.device = device

    def send(self, query, history):
        ans, _ = self.model.chat(
            self.tokenizer,
            query,
            history=history,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        return ans

    def send_stream(self, query, history):
        for i, (chunk_ans, _h, p_key) in enumerate(self.model.stream_chat(
            self.tokenizer,
            query,
            history=history,
            max_length=self.max_token,
            temperature=self.temperature
        )):
            yield chunk_ans

    @staticmethod
    def create_massages(query, history):
        messages = ""
        for dialog in history:
            messages += dialog[0]
            messages += dialog[1]
        messages += query
        print(messages)
        return messages

    def get_response(self, query, history):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        if self.streaming:
            # 暂时不用send_stream，输出逻辑上和本框架不符
            return self.send(query, history)
        else:
            return self.send(query, history)

    def change_model_name(self, model_name="chatglm-6b"):
        self.model_name = model_name
        self.load_model(model_name=model_name)

    def load_model(self,
                   **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.device == 'cuda':
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).half().cuda()
        else:
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).float()
        self.model = self.model.eval()

# Gemini模型
class GeminiLLM(BaseLLM):
    model: object = None
    model_name = 'gemini-pro'
    temperature = 0.1
    max_token = 1000
    streaming = False

    def __init__(self, temperature=0.1, max_token=1000, model_name='gemini-pro', streaming=False):
        self.set_para(temperature, max_token, model_name, streaming)
        self.load_model()

    def set_para(self, temperature, max_token, model_name, streaming):
        self.model_name = model_name
        self.temperature = temperature
        self.max_token = max_token
        self.streaming = streaming

    def send(self, query, history):
        messages = self.create_messages(query, history)
        response = self.model.generate_content(messages)
        return response.text

    def send_stream(self, query, history):
        messages = self.create_messages(query, history)
        response = self.model.generate_content(messages, stream=True)
        for chunk in response:
            yield chunk.text

    @staticmethod
    def create_messages(query, history):
        messages = []
        for dialog in history:
            messages.append({"role": "user", "parts": [dialog[0]]})
            messages.append({"role": "model", "parts": [dialog[1]]})
        messages.append({"role": "user", "parts": [query]})
        return messages

    def get_response(self, query, history):
        if self.streaming:
            return self.send_stream(query, history)
        else:
            return self.send(query, history)

    def load_model(self, **kwargs):
        try:
            from config_local import GOOGLE_API_KEY
        except ImportError:
            try:
                from config_example import GOOGLE_API_KEY
            except ImportError:
                raise ValueError("API key not found.")
        genai.configure(api_key=GOOGLE_API_KEY,transport='rest')
        self.model = genai.GenerativeModel(self.model_name)