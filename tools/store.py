from typing import List# 类型注解   

from langchain import FAISS# 向量存储

from tools.text import SimpleTextFilter, EntityTextFragmentFilter, fragment_text, \
    EntityVectorStoreFragmentFilter, SentenceSplitter, LineSplitter# 文本处理
from tools.utils import load_txt# 加载文本

# 将文档列表转换为字符串列表    
def docs_to_lst(docs):
    lst = []
    for e in docs:
        lst.append(e.page_content)
    return lst


# 将内存列表转换为字符串列表
def mem_to_lst(mem):
    lst = []
    for i in range(len(mem)):
        lst.append(mem[i].page_content)
    return lst


# 从向量存储中获取相关文本，返回文本列表
def get_related_from_vector_store(query, vs):
    related_text_with_score = vs.similarity_search_with_score(query)# vs.similarity_search_with_score(query)是向量存储的相似度搜索  
    docs = []
    for doc, score in related_text_with_score:
        docs.append(doc.page_content)
    return docs


# 整理实体名称，返回实体字典
def arrange_entities(entity_lst):
    entity_dicts = {}
    for entity in entity_lst:
        # 可能有中文冒号，统一替换为英文冒号
        entity = entity.replace('：', ':')
        k, v = entity.split(":", 1)# 1表示只分割一次
        entity_dicts[k] = v
    return entity_dicts


# 整理实体名称，返回实体名称列表
def arrange_entities_name(entity_lst):
    entity_names = []
    for entity in entity_lst:
        # 可能有中文冒号，统一替换为英文冒号
        entity = entity[0].replace('：', ':')
        k, v = entity[0].split(":", 1)# 1表示只分割一次
        entity_names.append(k)
    return entity_names


# 初始化向量存储，输入文本列表，返回向量存储或者None
def init_vector_store(embeddings,
                      filepath: str | List[str],
                      textsplitter):
    # docs = load_docs(filepath, textsplitter)
    txt = load_txt(filepath, textsplitter)# 加载文本,使用textsplitter分割文本   
    if len(txt) > 0:
        # vector_store = FAISS.from_documents(docs, embeddings)
        vector_store = FAISS.from_texts(txt, embeddings)# 使用embeddings向量存储文本，FAISS是向量存储的包来自langchain
        return vector_store
    else:
        return None


# 向量存储类
class VectorStore:
    # 初始化向量存储，输入embeddings,路径,文本分割器,chunk_size,top_k   
    def __init__(self, embeddings, path, textsplitter, chunk_size=20, top_k=6):
        self.top_k = top_k
        self.path = path
        self.core = init_vector_store(embeddings=embeddings, filepath=self.path, textsplitter=textsplitter)# 初始化向量存储
        self.chunk_size = chunk_size# 块大小

    # 相似度搜索，返回文本列表，core是向量存储
    def similarity_search_with_score(self, query):
        if self.core is not None:
            self.core.chunk_size = self.chunk_size
            return self.core.similarity_search_with_score(query, self.top_k)# 相似度搜索，core是向量存储的类来自langchain的FAISS
        else:
            return []

    # 获取路径
    def get_path(self):
        return self.path


# 简单存储类
class SimpleStoreTool:
    # 在初始化中，创建了文本分割器，实体过滤器，文本分割器，历史过滤器，事件过滤器，_k的参数是
    def __init__(self, info, entity_top_k, history_top_k, event_top_k):
        self.info = info
        self.ai_name = info.ai_name
        self.tsp = SentenceSplitter()# 文本分割器
        self.etfs = EntityTextFragmentFilter(tsp=self.tsp, top_k=history_top_k, entity_weight=0.8)# 实体过滤器

        self.entity_textsplitter = LineSplitter()# 实体文本分割器,LineSplitter是文本分割器的一种，用于将文本按行分割
        self.history_textsplitter = LineSplitter()# 历史文本分割器
        self.event_textsplitter = LineSplitter()# 事件文本分割器

        self.entity_text_filter = SimpleTextFilter(entity_top_k)# 实体文本过滤器,entity_top_k对于过滤器的作用是限制返回的文本数量    
        self.history_text_filter = SimpleTextFilter(history_top_k)# 历史文本过滤器
        self.event_text_filter = SimpleTextFilter(event_top_k)# 事件文本过滤器

    # 加载实体存储
    def load_entity_store(self):
        return load_txt(self.info.entity_path, self.entity_textsplitter)

    # 加载历史存储
    def load_history_store(self):
        return load_txt(self.info.history_path, self.history_textsplitter)

    # 加载事件存储
    def load_event_store(self):
        return load_txt(self.info.event_path, self.event_textsplitter)

    # 获取实体存储
    def get_entity_mem(self, query, store):
        return self.entity_text_filter.filter(query, store)

    # 获取历史存储
    def get_history_mem(self, query, store):
        return self.history_text_filter.filter(query, store)

    # 获取事件存储
    def get_event_mem(self, query, store):
        return self.event_text_filter.filter(query, store)

    # 实体片段
    def entity_fragment(self, query, entity_mem):
        entity_dict = arrange_entities(entity_mem)
        return self.etfs.filter(query, entity_dict)

    # 对话片段
    def dialog_fragment(self, query, dialog_mem):
        dialog_mem = fragment_text(dialog_mem, self.tsp)
        # 再次过滤
        dialog_mem = self.history_text_filter.filter(query, dialog_mem)
        for i, dialog in enumerate(dialog_mem):
            dialog_mem[i] = self.ai_name + '说：' + dialog
        return dialog_mem

    # 回答提取
    def answer_extract(self, mem, has_ai_name):
        # 提取对话，仅有ai的回答
        splitter = self.ai_name + '说：'
        for i, dialog in enumerate(mem):
            parts = dialog.split(splitter)
            mem[i] = splitter + parts[-1] if has_ai_name else parts[-1]


# 向量存储类
class VectorStoreTool:
    def __init__(self, info, embeddings, entity_top_k, history_top_k, event_top_k):
        self.info = info
        self.embeddings = embeddings
        self.ai_name = info.ai_name
        self.entity_top_k = entity_top_k
        self.history_top_k = history_top_k
        self.event_top_k = event_top_k

        self.ssp = SentenceSplitter()
        self.etfs = EntityVectorStoreFragmentFilter(tsp=self.ssp, top_k=entity_top_k, entity_weight=0.8)

        self.entity_textsplitter = LineSplitter()
        self.history_textsplitter = LineSplitter()
        self.event_textsplitter = LineSplitter()

    # 加载实体存储
    def load_entity_store(self):
        return VectorStore(self.embeddings,
                           self.info.entity_path,
                           top_k=self.entity_top_k,
                           textsplitter=self.entity_textsplitter)

    # 加载历史存储
    def load_history_store(self):
        return VectorStore(self.embeddings,
                           self.info.history_path,
                           top_k=self.history_top_k,
                           textsplitter=self.history_textsplitter)

    # 加载事件存储
    def load_event_store(self):
        return VectorStore(self.embeddings,
                           self.info.event_path,
                           top_k=self.event_top_k,
                           textsplitter=self.event_textsplitter)

    # 获取实体存储
    @staticmethod
    def get_entity_mem(query, store):
        return get_related_from_vector_store(query, store)

    # 获取历史存储
    @staticmethod
    def get_history_mem(query, store):
        return get_related_from_vector_store(query, store)

    # 获取事件存储
    @staticmethod
    def get_event_mem(query, store):
        return get_related_from_vector_store(query, store)

    # 实体片段
    def entity_fragment(self, query, entity_mem):
        entity_dict = arrange_entities(entity_mem)
        return self.etfs.filter(query, entity_dict, self.embeddings)

    # 对话片段
    def dialog_fragment(self, query, dialog_mem):
        dialog_mem = fragment_text(dialog_mem, self.ssp)
        # 再次过滤
        try:
            vs = FAISS.from_texts(dialog_mem, self.embeddings)
            dialog_with_score = vs.similarity_search_with_score(query, self.history_top_k)
        except IndexError:
            return []
        res_lst = []
        for doc in dialog_with_score:
            res_lst.append(self.ai_name + '说：' + doc[0].page_content)
        return res_lst

    # 回答提取
    def answer_extract(self, mem, has_ai_name):
        # 提取对话，仅有ai的回答
        splitter = self.ai_name + '说：'
        for i, dialog in enumerate(mem):
            parts = dialog.split(splitter)
            mem[i] = splitter + parts[-1] if has_ai_name else parts[-1]
