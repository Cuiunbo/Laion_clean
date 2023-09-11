#encoding:utf-8
'''
中文文本预处理函数，包含
* 繁简转换：繁体转简体
* 文档级过滤：过滤中文比例小、命名实体多、重复字符多的文档。
* 全半角转换：基于text-process的智能全半角转换
* 符号去除及转换：替换特殊字符，删除不必要的特殊字符
* 数据清洗：包含清洗使用错误的标点符号，删除首尾空格、删除汉字间的空格等
'''
import re
import functools
import multiprocessing
from tqdm import tqdm
from lxml import html


import opencc
from utils.utils import isChineseChar, isJapaneseChar, correctTitle
from utils.FilterData import FilterPara

t2s_convert=opencc.OpenCC('tw2s.json')

#
with open('config/bee.txt', 'r') as f:
    vocab = set(f.read().splitlines())

# 建立替换表
with open('config/change.txt') as f:
    mapping = {}
    for line in f:
        chars = line.strip('\n').split('\t')
        mapping[chars[0]] = chars[1]
    # 替换回车键至换行键
    mapping["\u000D"] = "\u000A"
    mapping["\u2028"] = "\u000A"
    mapping["\u2029"] = "\u000A"
    # 替换\t至空格
    mapping["\u0009"] = "\u0020"

with open("config/yiti_ch_change.txt") as f:
    yiti_change = {}
    for line in f:
        chars = line.strip('\n').split('\t')
        yiti_change[chars[0]] = chars[1]

def Fan2Jian(sent):
    '''
    繁简转换
    注意：某些 著 不会被转换
    '''
    s_sent_list=list(t2s_convert.convert(sent))

    # 修正不应该变但是会被错误改变的字符
    wrong_dict={'么':'幺'} 
    for correct_char,wrong_char in wrong_dict.items():
        if correct_char not in sent:
            continue
        pan_list=[]
        # 找出原句中不应该变以及会被错误改变的字符的顺序 （例如，繁体句中原来就出现的幺不能被替换为么）
        for x in sent:
            if x==correct_char:
                pan_list.append(True)
            elif x==wrong_char:
                pan_list.append(False)
        now=0
        for id,x in enumerate(s_sent_list):
            if x == wrong_char:
                if pan_list[now]:
                    s_sent_list[id]=correct_char
                now+=1
    s_sent=''.join(s_sent_list)
    # 修正不会被替换的        
    fix_dict={'甚么':'什么','妳':'你'}
    for x,y in fix_dict.items():
        s_sent=s_sent.replace(x,y)
    return s_sent

def CleanSent(str):
    '''
    数据清洗
    清洗使用错误的标点符号，删除首尾空格、删除汉字间的空格等
    '''

    def repl1(matchobj):
        return matchobj.group(0)[0]
    def repl2(matchobj):
        return matchobj.group(0)[-1]
    
    # 去掉段首尾换行
    str = str.strip("\n")
    # 标点重复
    str=re.sub(r'([（《【‘“\(\<\[\{）》】’”\)\>\]\} ,;:·；：、，。])\1+',repl1,str)
    # 括号紧跟标点
    str=re.sub(r'[（《【‘“\(\<\[\{][ ,.;:；：、，。！？·]',repl1,str)
    str=re.sub(r'[ ,.;:；：、，。！？·][）》】\)\>\]\}]',repl2,str)
    # 括号内为空
    str=re.sub(r'([（《【‘“\(\<\[\{\'\"][\'\"）》】’”\)\>\]\}])','',str)
    # 三个。和.以上的转为...
    str = re.sub(r'[。.]{3,}', '...', str)

    str = str.replace("\\", "")
    char_list=list(str) 
    # 删除句内汉字与其他符号之间的空格
    length = len(char_list)
    last = True
    for id, x in enumerate(char_list):
        if x==' ':
            if last or ((id+1<length)and(isChineseChar(char_list[id+1]) or char_list[id+1]==' ')):
                char_list[id]=''
            else:
                last = False
        else:
            if x in '。，：；！？【】《》“”':
                last = True
            else:
                last = isChineseChar(x)
    
    str = correctTitle(''.join(char_list))

    return str

def ReplaceNotation(str):
    '''
    替换特殊字符
    '''
    
    char_list = list(map(lambda x:mapping.get(x, x), str))
    
    for id, x in enumerate(char_list):
        if "\u2000" <= x <= "\u200F" or "\u0000" <= x <= "\u001F" and x != "\n":
            char_list[id]=''
    
    # 替换特殊字符
    return ''.join(char_list)

def ChangeYiti(str):
    return ''.join(map(lambda x:yiti_change.get(x, x), str))

def full2half(s: str) -> str:
    """
    Convert full-width characters to half-width ones.
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:  # 全角空格直接转换
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:  # 全角字符（除空格）根据关系转换
            num -= 0xfee0
        n.append(chr(num))
    return ''.join(n)

def remove_html(text):
    tree = html.fromstring(text)
    # def contains_html_tags(text):
    #     tree = html.fromstring(text)
    #     stripped_text = tree.text_content()
    #     return stripped_text != text
    # if contains_html_tags(text):
    #     print(f"contain tag : {text}")
    #     print(f"after : {tree.text_content()}")
    return tree.text_content()

def filter_language(text):
    whitespace_chars = set([" ", "\t", "\n", "\r", "\f", "\v", "\u00A0", "\u2028", "\u2029", "\u3000"])
    for char in text:
        if char in whitespace_chars:
            continue
        if char not in vocab:
            return  False
    return True

def Operate(content):
    content = ReplaceNotation(content)
    # content = remove_html(content)
    content = ChangeYiti(content)
    content = CleanSent(content)
    content = Fan2Jian(content)
    content = remove_html(content)
    return content

def SingleProcess(data_unit, is_filter):
    # 按照需求改写该部分函数

    if 'content' not in data_unit.keys():
        print(data_unit)
        return None

    if not data_unit['content']:
        return None
        
    # 章节格式
    if data_unit['content'] and isinstance(data_unit['content'][0], dict):
        if is_filter:
            for item in range(len(data_unit['content'])-1, -1, -1):
                if len(''.join(data_unit['content'][item]['sub_content'])) <= len(data_unit['content'][item]['sub_content']):
                    del data_unit['content'][item]
                    continue
                if not FilterPara(''.join(data_unit['content'][item]['sub_content'])):
                    del data_unit['content'][item]
            
        full_text = ""
        for i, content in enumerate(data_unit['content']):
            data_unit['content'][i]['sub_title'] = Operate(content['sub_title'])
            data_unit['content'][i]['sub_content'] = [Operate(sub_content) for sub_content in content['sub_content']]
        #     full_text += ''.join(content['sub_content'])
        # if len(full_text) < 32:
        #     return None

        if is_filter:
            for item in range(len(data_unit['content'])-1, -1, -1):
                if len(''.join(data_unit['content'][item]['sub_content'])) <= len(data_unit['content'][item]['sub_content']):
                    del data_unit['content'][item]
                    continue
                if not FilterPara(''.join(data_unit['content'][item]['sub_content'])):
                    del data_unit['content'][item]

    else: # 一般文章格式
        if isinstance(data_unit['content'], str):
            data_unit['content'] = [data_unit['content']]
        if is_filter and not FilterPara(''.join(data_unit['content'])):
            return None
        data_unit['content'] = [Operate(content) for content in data_unit['content']]
        if is_filter and not FilterPara(''.join(data_unit['content'])):
            return None

    if not data_unit["content"]:
        return None

    if 'meta' in data_unit:
        if 'abstract' in data_unit['meta']:
            data_unit['meta']['abstract'] = [Operate(abstract) for abstract in data_unit['meta']['abstract']]
        if 'keywords' in data_unit['meta']:
            data_unit['meta']['abstract'] = [Operate(keyword) for keyword in data_unit['meta']['keywords']]


    return data_unit

def MultiProcess(ds, num_proc):
    results = []
    with multiprocessing.Pool(num_proc) as p:
        max_ = len(ds)
        with tqdm(total=max_, desc='Operating data') as pbar:
            for message in enumerate(p.imap_unordered(functools.partial(SingleProcess), ds)):
                # 对返还结果做处理
                if message:
                    results.append(message)
                pbar.update()
    return results

def main():
    pass
    # reserve, MAPPING = CreateRM()
    # # 清洗文本
    # print(Operate(text, reserve, MAPPING))
    # # 清洗单个json数据
    # SingleProcess(data_unit, reserve, MAPPING)
    # # 清洗json列表
    # MultiProcess(data_list, reserve, MAPPING, num_proc=10)

if __name__ =="__main__":
    main()