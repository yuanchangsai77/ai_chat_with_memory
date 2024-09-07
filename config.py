import configparser

import openai

model_map = {
    'chatglm-6b-int4': 'ChatGLMLLM',
    'chatglm-6b': 'ChatGLMLLM',
    'THUDM/chatglm2-6b-int4': 'ChatGLMLLM',
    'chatglm2-6b-int4': 'ChatGLMLLM',
    'gpt-3.5-turbo': 'GPT3_5LLM',
    'gpt-3.5-turbo-0613': 'GPT3_5LLM',
    'gemini-pro': 'GeminiLLM'  # 新增Gemini模型映射
}# 模型映射

# 基础配置
class BaseConfig:

    def __init__(self):
        self.file = configparser.ConfigParser()# 配置文件

        self.file.read('config.ini', encoding='utf-8-sig')
        try:
            openai.api_key = self.file.get('API', 'openai_api_key')
        except:
            pass
        # 身份
        self.ai_name = self.file['SINGLE_AI']['ai_name']
        self.world_name = self.file['SINGLE_AI']['world_name']
        self.user_name = self.file['SINGLE_AI']['user_name']
        # 多人
        ai_names = self.file['MULTI_AI']['ai_names']
        # 获取名字列表
        valid_punctuation = ['、', '，', '.', '。', '|', '/']# 有效标点
        for p in valid_punctuation:
            ai_names = ai_names.replace(p, ',')
        ai_names = ai_names.split(',')
        self.ai_names = [name for name in ai_names if name]
        self.multi_ai_world_name = self.file['MULTI_AI']['world_name']
        self.first_ai = self.file['MULTI_AI']['first']
        self.greeting = self.file['MULTI_AI']['greeting']
        self.auto = self.file.getboolean('MULTI_AI', 'auto')
        self.delay_s = self.file.getint('MULTI_AI', 'delay') if self.auto else 0
        self.multi_agent_chat_strategy = self.file.get('MULTI_AI', 'strategy')
        # 记忆
        self.lock_memory = self.file.getboolean('MEMORY', 'lock_memory')
        self.history_window = self.file.getint('MEMORY', 'history_window')
        self.entity_top_k = self.file.getint('MEMORY', 'entity_top_k')
        self.history_top_k = self.file.getint('MEMORY', 'history_top_k')
        self.event_top_k = self.file.getint('MEMORY', 'event_top_k')
        # 历史
        self.dialog_max_token = self.file.getint('HISTORY', 'dialog_max_token')
        self.window_max_token = self.file.getint('HISTORY', 'window_max_token')
        self.token_decrease = self.file.getint('HISTORY', 'token_decrease')
        # 声音
        self.voice_enabled = self.file.getboolean('VOICE', 'enabled')
        self.speak_rate = self.file['VOICE']['speak_rate']
        # 输出
        self.streaming = self.file.getboolean('OUTPUT', 'streaming')
        self.words_per_line = self.file.getint('OUTPUT', 'words_per_line')
        # 模型
        self.model_name = self.file['MODEL']['model_name']
        self.LLM = model_map[self.model_name]
        self.temperature = self.file.getfloat('MODEL', 'temperature')
        self.model_device = self.file['MODEL']['model_device']
        self.use_embedding_model = self.file.getboolean('MODEL', 'use_embedding_model')
        self.embedding_model = self.file['MODEL']['embedding_model']
        self.embedding_model_device = self.file['MODEL']['embedding_model_device']

    # 保存到文件
    def save_to_file(self):
        with open('config.ini', 'w', encoding='utf-8') as configfile:
            self.file.write(configfile)

    # 设置ai名称
    def set_ai_name(self, ai_name):
        self.ai_name = ai_name
        self.file.set('SINGLE_AI', 'ai_name', ai_name)

    # 设置世界名称
    def set_world_name(self, world_name):
        self.world_name = world_name
        self.file.set('SINGLE_AI', 'world_name', world_name)

    # 设置用户名称
    def set_user_name(self, user_name):
        self.user_name = user_name
        self.file.set('SINGLE_AI', 'user_name', user_name)
    
    # 设置锁定记忆
    def set_lock_memory(self, lock_memory):
        self.lock_memory = lock_memory
        self.file.set('MEMORY', 'lock_memory', str(lock_memory))

    # 设置历史窗口
    def set_history_window(self, history_window):
        self.history_window = history_window
        self.file.set('MEMORY', 'history_window', str(history_window))

    # 设置窗口最大token
    def set_window_max_token(self, window_max_token):
        self.window_max_token = window_max_token
        self.file.set('HISTORY', 'window_max_token', str(window_max_token))
    
    # 设置对话最大token
    def set_dialog_max_token(self, dialog_max_token):
        self.dialog_max_token = dialog_max_token
        self.file.set('HISTORY', 'dialog_max_token', str(dialog_max_token))

    # 设置token减少
    def set_token_decrease(self, token_decrease):
        self.token_decrease = token_decrease
        self.file.set('HISTORY', 'token_decrease', str(token_decrease))

    # 设置实体最大k
    def set_entity_top_k(self, entity_top_k):
        self.entity_top_k = entity_top_k
        self.file.set('MEMORY', 'entity_top_k', str(entity_top_k))

    # 设置历史最大k
    def set_history_top_k(self, history_top_k):
        self.history_top_k = history_top_k
        self.file.set('MEMORY', 'history_top_k', str(history_top_k))

    # 设置事件最大k
    def set_event_top_k(self, event_top_k):
        self.event_top_k = event_top_k
        self.file.set('MEMORY', 'event_top_k', str(event_top_k))

    # 设置流式输出
    def set_streaming(self, streaming):
        self.streaming = streaming
        self.file.set('OUTPUT', 'streaming', str(streaming))

    # 设置温度
    def set_temperature(self, temperature):
        self.temperature = temperature
        self.file.set('MODEL', 'temperature', str(temperature))


# 开发配置
class DevConfig:
    # 初始化
    def __init__(self):
        self.file = configparser.ConfigParser()# 配置文件
        self.file.read('dev_settings.ini', encoding='utf-8-sig')
        self.word_similarity_threshold = self.file.getfloat('TEXT', 'word_similarity_threshold')# 单词相似度阈值
        self.update_history_store_step = self.file.getint('TEXT', 'update_history_store_step')# 更新历史存储步骤
        self.similarity_comparison_context_window = \
            self.file.getint('TEXT', 'similarity_comparison_context_window')# 相似度比较上下文窗口
        self.answer_extract_enabled = self.file.getboolean('TEXT', 'answer_extract_enabled')# 回答提取是否启用
        self.fragment_answer = self.file.getboolean('TEXT', 'fragment_answer')# 回答片段
        self.openai_text_moderate = self.file.getboolean('MODERATION', 'openai_text_moderate')# 开放ai文本审查
        # ------
        self.DEBUG_MODE = self.file.getboolean('TEXT', 'DEBUG_MODE')# 调试模式

    # 保存到文件
    def save_to_file(self):
        with open('dev_settings.ini', 'w', encoding='utf-8') as configfile:
            self.file.write(configfile)

    # 设置单词相似度阈值
    def set_word_similarity_threshold(self, word_similarity_threshold):
        self.word_similarity_threshold = word_similarity_threshold
        self.file.set('TEXT', 'word_similarity_threshold', str(word_similarity_threshold))# 设置单词相似度阈值

    # 设置更新历史存储步骤
    def set_update_history_store_step(self, update_history_store_step):
        self.update_history_store_step = update_history_store_step
        self.file.set('TEXT', 'update_history_store_step', str(update_history_store_step))# 设置更新历史存储步骤

    # 设置相似度比较上下文窗口
    def set_similarity_comparison_context_window(self, similarity_comparison_context_window):
        self.similarity_comparison_context_window = similarity_comparison_context_window
        self.file.set('TEXT', 'similarity_comparison_context_window', str(similarity_comparison_context_window))# 设置相似度比较上下文窗口

    # 设置回答提取是否启用
    def set_answer_extract_enabled(self, answer_extract_enabled):
        self.answer_extract_enabled = answer_extract_enabled
        self.file.set('TEXT', 'answer_extract_enabled', str(answer_extract_enabled))# 设置回答提取是否启用

    # 设置回答片段
    def set_fragment_answer(self, fragment_answer):
        self.fragment_answer = fragment_answer
        self.file.set('TEXT', 'fragment_answer', str(fragment_answer))# 设置回答片段

    # 设置调试模式
    def set_debug_mode(self, debug_mode):
        self.DEBUG_MODE = debug_mode
        self.file.set('TEXT', 'DEBUG_MODE', str(debug_mode))# 设置调试模式
