import importlib# 导入的importlib模块是用于动态加载模块的
import time# 导入的time模块是用于获取当前时间的
from typing import List# 导入的List模块是用于类型提示的

from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

from agent import MainAgent# 导入的MainAgent模块是用于创建主代理的
from command import command_cleanup_task, Pool, command_flags, execute_command, debug_msg_pool# 导入的command模块是用于处理命令的
from tools.utils import delete_last_line# 导入的utils模块是用于删除最后一行的


def get_class(module_name, class_name):# 获取类的函数
    module = importlib.import_module(module_name)# 动态加载模块
    return getattr(module, class_name)# 获取类

# 创建llm的函数
def create_llm(config):
    llm_class = get_class("agent.llm", config.LLM)# 获取llm类
    llm_instance = llm_class(temperature=config.temperature,
                             max_token=config.dialog_max_token,
                             model_name=config.model_name,
                             streaming=config.streaming)
    if hasattr(llm_instance, 'load_model'):# 如果llm类有load_model方法
        llm_instance.load_model()# 加载模型
    if hasattr(llm_instance, 'set_device'):# 如果llm类有set_device方法
        llm_instance.set_device(config.model_device)# 设置设备
    return llm_instance# 返回llm实例

# 创建huggingface嵌入模型的函数
def create_huggingface_embedding_model(config):
    return HuggingFaceEmbeddings(model_name=config.embedding_model,
                                 model_kwargs={'device': config.embedding_model_device})# 返回huggingface嵌入模型

# 创建openai嵌入模型的函数
def create_openai_embedding_model():
    return OpenAIEmbeddings()# 返回openai嵌入模型

# 沙盒类
class Sandbox:

    def __init__(self, world_name):# 初始化函数
        self.world_name = world_name
        self.chat_str = ''
        self.default_user_name = ''
        self.one_agent = True
        self.use_embed_model = True
        self.language_model = None
        self.embedding_model = None
        self.multi_agent_chat_strategy = 'round'
        self.ai_names = None
        self.cur_ai = ''
        self.cur_agent = None
        self.auto = False
        self.delay_s = 10
        self.tmp_msg = ''

    # 设置当前代理的身份
    def cur_agent_set_identity(self, world_name, ai_name, user_name):
        self.cur_agent.set_identity(world_name, ai_name, user_name)
        # 重要：切换了角色后指令池要重置
        command_flags.reset()
    # 重新加载配置
    def cur_agent_reload_config(self, config):
        self.cur_agent.reload_config(config)
    # 重新加载开发配置
    def cur_agent_reload_dev_config(self):
        self.cur_agent.reload_dev_config()
    # 获取临时消息
    def get_tmp_message(self):
        return self.tmp_msg
    # 设置模型
    def set_models(self, config):
        self.language_model = create_llm(config)

        if config.use_embedding_model:
            if config.embedding_model == 'openai':
                self.embedding_model = create_openai_embedding_model()
            else:
                self.embedding_model = create_huggingface_embedding_model(config)
    # 初始化全局代理
    def init_global_agent(self, config):
        self.one_agent = True
        self.set_models(config)
        self.cur_agent = self.init_one_agent(config, config.world_name, config.ai_name)
        self.default_user_name = self.cur_agent.user_name
    # 初始化一个代理
    def init_one_agent(self, config, world_name, ai_name):
        return MainAgent(world_name, ai_name, self.language_model, self.embedding_model, config)
    # 聊天
    def chat(self, chat_str):
        self.chat_str = chat_str
        self.tmp_msg = self.check_command(chat_str, self.cur_agent)
        if self.tmp_msg == 'chat':
            return self.cur_agent.chat(self.chat_str)
        else:
            return None
        
    # 多代理聊天
    def chat_with_multi_agent(self, config):
        self.ai_names = config.ai_names
        self.cur_ai = config.first_ai
        self.chat_str = config.greeting
        self.multi_agent_chat_strategy = config.multi_agent_chat_strategy
        self.set_models(config)

        agents = []
        for name in config.ai_names:
            agents.append(self.init_one_agent(config, world_name=config.world_name, ai_name=name))

        if self.multi_agent_chat_strategy == 'round':
            self.round_chat(agents)
        else:
            print("策略参数错误,请检查config.ini")
            debug_msg_pool.append_msg("策略参数错误,请检查config.ini")
    # 轮询聊天
    def round_chat(self, agents: List):
        idx = self.ai_names.index(self.cur_ai)
        ai_num = len(self.ai_names)
        # 处理首个对话，加入到该ai的历史中
        agents[idx].save_dialog_to_file(self.cur_ai + '说：' + self.chat_str + '\n')
        # 重新加载历史
        agents[idx].load_history(agents[idx].basic_history)

        # 开始轮询对话
        while True:
            idx = (idx + 1) % ai_num
            agents[idx].set_user_name(self.cur_ai)
            self.chat_str = agents[idx].chat(self.chat_str)
            self.cur_ai = agents[idx].ai_name
            if self.auto:
                time.sleep(self.delay_s)
            else:
                input_str = ' '
                while input_str != '':
                    input_str = input("按回车继续")
    # 检查命令
    def check_command(self, query, agent):
        # ------指令部分
        # 指令收尾工作
        if command_flags.continue_talk:
            agent.set_user_name(self.default_user_name)
        command_cleanup_task(agent)
        # 检查是否为指令
        Pool().check(query, agent.ai_name)
        if not command_flags.not_command:
            sys_mes = execute_command(agent)
            # 执行重试指令
            if command_flags.retry:
                if agent.query == '':
                    print("当前没有提问，请输入提问。")
                    debug_msg_pool.append_msg("当前没有提问，请输入提问。")
                    return 'ai_chat_with_memory sys:当前没有提问，无法重试提问。'
                # 从临时存储中取出提问
                self.chat_str = agent.get_tmp_query()
                if not agent.base_config.lock_memory:
                    # 删除历史文件最后一行
                    delete_last_line(agent.info.history_path)
                    # 重新加载临时历史对话
                    agent.load_history(agent.basic_history)
                else:
                    # 若为锁定记忆模式，需要删除历史记录最后一次对话
                    agent.history.pop()
                agent.step -= 1
            # 执行“继续说”指令
            elif command_flags.continue_talk:
                last_ans = agent.last_ans
                self.chat_str = last_ans
                if self.chat_str == '':
                    # 无最后的回答缓存，加载最后的历史对话
                    splitter = agent.ai_name + '说'
                    ans_str = agent.history[-1][1]
                    self.chat_str = ans_str[ans_str.find(splitter) + len(splitter):].replace('\n', '')
                agent.set_user_name(agent.ai_name)
            elif command_flags.exit:
                return 'exit'
            else:
                return 'ai_chat_with_memory sys:执行了指令。'
        elif command_flags.wrong_command:
            return 'ai_chat_with_memory sys:错误指令。'
        return 'chat'
        # ------
