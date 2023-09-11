import collections
import jieba.analyse
from utils.utils import countChineseChar, countChineseChar, isChineseWord, countJapaneseChar, countTraditionalChar, countEnglishChar
import re

f = open("config/rare_ch_char.txt", encoding='utf-8')
rare = set(f.read())

def if_dup(para): # continuous 3 repeat n-grams
    for pos in range(len(para)):
        l_len = min(15, pos)
        r_len = min(15, len(para) - pos)
        curr_len = min(l_len, r_len)
        for i in range(2, curr_len):
            l_word = para[pos - i: pos]
            r_word = para[pos: pos + i]
            if l_word == r_word:
                if pos + 2 * i <= len(para) and para[pos + i : pos + 2 * i] == l_word and isChineseWord(l_word):
                    return True
    return False

def if_rare_str(para):
    cnt = 0
    tmp_str = ''
    if not para:
        return True
    for ch in para:
        if ch in rare:
            cnt += 1
            tmp_str += ch
        else:
            if cnt >= 10:
                return True
            cnt = 0
            tmp_str = ''
    if cnt >= 10:
        return True
    return False

def if_bad_para(para):
    limit = 25
    cnt = 0
    word_list = jieba.cut(para, HMM=False)
    tmp_str = ''
    if not para:
        return True
    for ch in word_list:
        if len(ch) == 1:
            cnt += 1
            tmp_str += ch
        else:
            if cnt >= limit:
                return True
            cnt = 0
            tmp_str = ''
    if cnt >= limit:
        return True
    return False

def FilterPara(para):
    '''
    文本过滤
    根据中文字数及比例、长度对句子进行过滤
    '''

    length = len(para)
    
    # 过滤字符数
    # for i in range(0, length, 512):
    #     if collections.Counter(list(para[0+i:512+i])).most_common(1)[0][1] > 40:
    #         print(f"common :[]>>{para}")
    #         return False

    if if_rare_str(para) or if_dup(para) or if_bad_para(para):
        print(f"rare :[]>>{para}, rare{if_rare_str(para)},dup{if_dup(para)} or bad{if_bad_para(para)}")
        return False

    # 命名实体识别
    kw = jieba.analyse.extract_tags(para, allowPOS=('ns', 'nr', 'nt', 'nw', 'nz'))

    if len(''.join(kw))/length > 0.5:
        print(f"kw too often :[]>>{para}")
        return False
    
    return True
