import ast# 解析字符串
import copy# 深拷贝
import textdistance# 文本距离
from langchain import FAISS# 向量存储
import re# 正则表达式
from typing import List# 类型注解

# 回答片段文本分割器
class AnswerFragmentTextSplitter:

    def split_text(self, text_lst: List[str], text_splitter) -> List[str]:
        all_text = ''
        for text in text_lst:
            all_text += text
        new_lst = text_splitter.split_text(all_text)# 将所有文本拼接成一个字符串，然后使用text_splitter分割成多个字符串
        # 移除空字符串
        lst = [e for e in new_lst if e]
        return lst

# 行分割器
class LineSplitter:

    @staticmethod# 静态方法，不需要实例化就可以调用
    def split_text(text) -> List[str]:
        words = text.split('\n')# 将文本按行分割
        # 去掉空白字符串
        words = [word for word in words if word]
        return words

# 句子分割器
class SentenceSplitter:

    def split_text(self, text) -> List[str]:
        # 中英文标点符号
        punctuations = r'''！？。；!?.;'''
        # words = re.split('[' + punctuations + ']', text)
        words = re.findall(f'[^{punctuations}]*[{punctuations}]?', text)# 使用正则表达式分割文本

        # 去掉空白字符串
        words = [word for word in words if word]
        return words

# 文本碎片化
def fragment_text(text_lst, text_splitter):
    return AnswerFragmentTextSplitter().split_text(text_lst, text_splitter)

# 高相似度文本过滤器
def high_word_similarity_text_filter(agent, mem_lst):
    # 字词相似度比对，算法复杂度o(n^2)
    remaining_memory = copy.deepcopy(mem_lst)  # 创建一个副本以避免在迭代时修改原始列表
    for i in range(len(mem_lst)):
        for j in range(len(mem_lst)):
            if i != j:
                str_i = mem_lst[i]
                str_j = mem_lst[j]
                sim_score = textdistance.jaccard(str_i, str_j)# 计算两个字符串的字词相似度使用jaccard算法
                if sim_score > agent.dev_config.word_similarity_threshold:
                    # 如果两个字符串的字词相似度超过阈值，则删除较短的字符串（较短的字符串信息含量大概率较少）
                    del_e = mem_lst[i] if len(str_i) < len(str_j) else mem_lst[j]
                    if del_e in remaining_memory:
                        remaining_memory.remove(del_e)
    return remaining_memory

# 边界文本过滤器
class BoundTextFilter:

    def __init__(self, interval_str):
        self.lower_bound_type = 'closed' if interval_str[0] == '[' else 'open'# 如果区间字符串的第一个字符是[，则下界为闭区间，否则为开区间
        self.upper_bound_type = 'closed' if interval_str[-1] == ']' else 'open'# 如果区间字符串的最后一个字符是]，则上界为闭区间，否则为开区间
        interval = ast.literal_eval(interval_str.strip('[]()'))# 使用ast.literal_eval函数将区间字符串转换为区间列表
        self.lower_bound = interval[0]# 下界
        self.upper_bound = interval[1]# 上界

    # 比较函数
    def _compare(self, upper_bound: bool, open_bound: bool, num: float):
        if upper_bound:
            return num < self.upper_bound if open_bound else num <= self.upper_bound# 如果上界为闭区间，则返回num小于上界，否则返回num小于等于上界
        else:
            return num > self.lower_bound if open_bound else num >= self.lower_bound# 如果下界为闭区间，则返回num大于下界，否则返回num大于等于下界

    # 比较函数，判断是否在区间内
    def compare(self, num):
        
        lob = self.lower_bound_type == 'open'# 下界是否为开区间
        uob = self.upper_bound_type == 'open'# 上界是否为开区间
        return self._compare(upper_bound=False, open_bound=lob, num=num) \
            and self._compare(upper_bound=True, open_bound=uob, num=num)# 如果num在区间内，则返回True，否则返回False

    # 过滤函数，过滤掉不在区间内的文本
    def filter(self, query, docs):
        mem_lst = []
        for t in docs:
            sim_score = textdistance.jaccard(query, t.page_content)# 计算文本相似度
            if self.compare(sim_score):# 如果相似度在区间内，则将文本添加到mem_lst中
                mem_lst.append(t)
        return mem_lst

# 栈
class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        else:
            return None

    def top(self):
        if not self.is_empty():
            return self.stack[-1]
        else:
            return None

    def clear(self):
        self.stack.clear()

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)

    def get_lst(self):
        return self.stack

# 简单文本过滤器
class SimpleTextFilter:

    def __init__(self, top_k):
        self.top_k = top_k

    # 过滤函数，过滤掉相似度不在前top_k的文本
    def filter(self, query, docs):
        min_num_stack = Stack()# 创建一个栈
        tmp_stack = Stack()# 创建一个临时栈
        min_num_stack.push({'text': '', 'score': 1.0})# 将一个空文本和相似度1.0压入栈中
        for t in docs:
            sim_score = textdistance.ratcliff_obershelp(query, t)# 计算文本相似度
            if sim_score > min_num_stack.top()['score']:# 如果当前相似度大于栈顶元素的相似度
                # 只要当前相似度大于栈顶元素的相似度，则将栈顶元素弹出并压入tmp_stack中，并将当前文本和相似度压入栈中
                while sim_score > min_num_stack.top()['score']:
                    tmp_stack.push(min_num_stack.pop())
                min_num_stack.push({'text': t, 'score': sim_score})
                # 只要tmp_stack不为空且min_num_stack中的元素数量小于top_k，则将tmp_stack中的元素弹出并压入min_num_stack中
                while not tmp_stack.is_empty() and min_num_stack.size() - 1 < self.top_k:
                    min_num_stack.push(tmp_stack.pop())

                tmp_stack.clear()
            else:# 如果当前相似度小于栈顶元素的相似度
                if min_num_stack.size() - 1 < self.top_k:# 如果min_num_stack中的元素数量小于top_k，则将当前文本和相似度压入栈中
                    min_num_stack.push({'text': t, 'score': sim_score})
        if min_num_stack.size() > 1:# 如果min_num_stack中的元素数量大于1，则将栈中的元素弹出并返回

            final_lst = []
            stack_lst = min_num_stack.get_lst()[1:]
            for e in stack_lst:
                final_lst.append(e['text'])
            return final_lst
        else:
            return []

# 实体文本碎片化过滤器
class EntityTextFragmentFilter:

    def __init__(self, top_k, tsp, entity_weight=0.1):
        self.tsp = tsp
        self.top_k = top_k
        self.entity_weight = entity_weight

    # 给实体名相似程度进行打分
    @staticmethod
    def get_dict_with_scores(query, entity_dict):
        new_dict = {}
        # 对实体名相似程度进行打分
        for name, describe in entity_dict.items():
            new_dict[name] = {}
            new_dict[name]['score'] = textdistance.ratcliff_obershelp(query, name)# 计算实体名相似度
            new_dict[name]['text'] = describe# 实体描述

        return new_dict

    # 过滤函数
    def filter(self, query, entity_dict):
        # ---给实体相似度打分
        # entity_dict_with_score = self.get_dict_with_scores(query, entity_dict)
        # ---
        # ---打碎实体描述文本
        # entity_mem = []
        # for name, tur in entity_dict_with_score.items():
        #     # 实体打碎策略：先切分，后罗列
        #     describe_lst = self.tsp.split_text(tur['text'])  # 切分实体描述
        #     for d in describe_lst:  # 罗列
        #         entity_mem.append({'text': name + ':' + d, 'entity_score': tur['score']})
        # # ---

        # ---打碎实体描述文本
        entity_mem = []
        for name, describe in entity_dict.items():
            # 实体打碎策略：先切分，后罗列
            describe_lst = self.tsp.split_text(describe)  # 切分实体描述
            for d in describe_lst:  # 罗列
                entity_mem.append(name + ':' + d)
        # ---

        min_num_stack = Stack()
        tmp_stack = Stack()
        min_num_stack.push({'text': '', 'score': 1.0})
        for e in entity_mem:

            # # 计算方式：描述相似度 * 字符串权重 + 实体相似度 * 实体权重
            # lcs_sim_score = textdistance.ratcliff_obershelp(query, e['text'])
            # sim_score = lcs_sim_score * (1 - self.entity_weight) + e['entity_score'] * self.entity_weight

            sim_score = textdistance.ratcliff_obershelp(query, e)

            if sim_score > min_num_stack.top()['score']:
                while sim_score > min_num_stack.top()['score']:
                    tmp_stack.push(min_num_stack.pop())
                min_num_stack.push({'text': e, 'score': sim_score})
                while not tmp_stack.is_empty() and min_num_stack.size() - 1 < self.top_k:
                    min_num_stack.push(tmp_stack.pop())
                tmp_stack.clear()
            else:
                if min_num_stack.size() - 1 < self.top_k:
                    min_num_stack.push({'text': e, 'score': sim_score})
        if min_num_stack.size() > 1:
            final_lst = []
            stack_lst = min_num_stack.get_lst()[1:]
            for e in stack_lst:
                final_lst.append(e['text'])
            return final_lst
        else:
            return []

# 实体向量存储碎片化过滤器
class EntityVectorStoreFragmentFilter:

    def __init__(self, top_k, tsp, entity_weight=0.8):
        self.tsp = tsp
        self.top_k = top_k
        self.entity_weight = entity_weight

    # 获取实体名
    @staticmethod# 静态方法，不需要实例化就可以调用
    def get_entity_names(entity_mem):
        entity_names = []
        for entity in entity_mem:
            # 可能有中文冒号，统一替换为英文冒号
            entity_name = entity.replace('：', ':')
            k, v = entity_name.split(":", 1)
            entity_names.append(k)
        return entity_names

    # 过滤函数
    def filter(self, query, entity_dict, embeddings):
        # ---打碎实体描述文本
        entity_mem = []
        for name, describe in entity_dict.items():
            # 实体打碎策略：先切分，后罗列
            describe_lst = self.tsp.split_text(describe)  # 切分实体描述
            for d in describe_lst:  # 罗列
                entity_mem.append(name + ':' + d)
        # ---
        vs = FAISS.from_texts(entity_mem, embeddings)
        entity_with_score = vs.similarity_search_with_score(query, self.top_k)
        res_lst = []
        for doc in entity_with_score:
            res_lst.append(doc[0].page_content)

        return res_lst
