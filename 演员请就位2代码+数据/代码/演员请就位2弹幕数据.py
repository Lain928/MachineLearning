# 导入库
import os  
import jieba
import numpy as np
import pandas as pd 

from pyecharts.charts import Bar, Pie, Line, WordCloud, Page
from pyecharts import options as opts 
from pyecharts.globals import SymbolType, WarningType
WarningType.ShowWarning = False

import stylecloud
from IPython.display import Image # 用于在jupyter lab中显示本地图

# 读入数据
data_list = os.listdir('../data/')

df_all = pd.DataFrame()

for i in data_list[-10:]:
    print(i) 
    df_one = pd.read_csv(f'../data/{i}', engine='python', encoding='utf-8', index_col=0) 
    df_all = df_all.append(df_one, ignore_index=False)

print(df_all.shape) 

df_all.info() 


# 删除弹幕角色
df_all['content'] = df_all['content'].str.replace('(.*?:)', '')
df_all.head() 

df_epinum

df_epinum = df_all['episodes'].value_counts().reset_index()
df_epinum['num'] = [1, 5, 3, 7, 6, 8, 4, 9, 2, 10]
df_epinum = df_epinum.sort_values('num') 
df_epinum

x_data = df_epinum['index'].tolist()
y_data = df_epinum['episodes'].tolist()

# 条形图
bar1 = Bar(init_opts=opts.InitOpts(width='1350px', height='750px'))
bar1.add_xaxis(xaxis_data=x_data)
bar1.add_yaxis('', y_axis=y_data)
bar1.set_global_opts(title_opts=opts.TitleOpts(title='前五期的弹幕数走势图'), 
                     visualmap_opts=opts.VisualMapOpts(max_=60000, is_show=False) 
                    )
bar1.render() 

chen_num = df_all.content.str.contains('凯歌').sum()
er_num = df_all.content.str.contains('冬升').sum()
zhao_num = df_all.content.str.contains('赵薇').sum()
guo_num = df_all.content.str.contains('敬明|小四').sum()
li_num = df_all.content.str.contains('诚儒').sum()

print(chen_num, er_num, zhao_num, guo_num, li_num) 


df_num = pd.DataFrame({
    'name': ['陈凯歌', '尔冬升', '赵薇', '郭敬明', '李诚儒'],
    'number': [chen_num, er_num, zhao_num, guo_num, li_num]
})
df_num = df_num.sort_values('number', ascending=False)
df_num



# 产生数据
x_data = df_num['name'].values.tolist()
y_data = df_num['number'].values.tolist()

# 条形图
bar2 = Bar(init_opts=opts.InitOpts(width='1350px', height='750px'))
bar2.add_xaxis(x_data)
bar2.add_yaxis('', y_data)
bar2.set_global_opts(title_opts=opts.TitleOpts(title='弹幕中主要导演的提及次数'), 
                     visualmap_opts=opts.VisualMapOpts(max_=int(max(df_num['number'])), is_show=False)
                    )

bar2.render() 


actor = '小彩旗 / 曹骏 / 丁程鑫 / 董思怡 / 费启鸣 / 郭晓婷 / 贺开朗 / 黄璐 / 黄梦莹 / 胡杏儿 / 黄奕 / 辣目洋子 / 刘芮麟 / 李溪芮 / 娄艺潇 / 李智楠 / 马伯骞 / 马苏 / 孟子义 / 倪虹洁 / 任敏 / 施柏宇 / 孙千 / 孙阳 / 唐一菲 / 陈宥维 / 何昶希 / 王楚然 / 王锵 / 王莎莎 / 王智 / 温峥嵘 / 晏紫东 / 杨志刚 / 张大大 / 张海宇 / 张铭恩 / 张月 / 张逸杰 / 邹元清 / 李诚儒 / 尹子维 / 王茂蕾 / 秦越 / 张熙然 / 李彩桦 / 沈保平 / 马志威'
actor_list = actor.split(' / ')
actor_list[:5] 

tiji_num = [df_all.content.str.contains(i).sum() for i in actor_list] 
tiji_num[:5]  

df_actor = pd.DataFrame({
    'actor_name': actor_list,
    'tiji_num': tiji_num
})

df_actor.head() 

tiji_top10 = df_actor.sort_values('tiji_num', ascending=False).head(10)
tiji_top10 = tiji_top10.sort_values('tiji_num')
tiji_top10
# 产生数据
x_data = tiji_top10['actor_name'].values.tolist()
y_data = tiji_top10['tiji_num'].values.tolist()

# 条形图
bar3 = Bar(init_opts=opts.InitOpts(width='1350px', height='750px'))
bar3.add_xaxis(x_data)
bar3.add_yaxis('', y_data)
bar3.set_global_opts(title_opts=opts.TitleOpts(title='弹幕中演员提及次数排行Top10'), 
                     visualmap_opts=opts.VisualMapOpts(max_=int(max(tiji_top10['tiji_num'])), is_show=False)
                    )
bar3.set_series_opts(label_opts=opts.LabelOpts(position='right'))
bar3.reversal_axis()
bar3.render() 

page = Page() 
page.add(bar1, bar2, bar3)
page.render('../image/演员请就位2弹幕分析.html')

def get_cut_words(x_series):
    # 读入停用词表
    stop_words = [] 
    
    with open(r"C:\Users\wzd\Desktop\CDA\CDA_Python\Python文本分析\10.文本摘要\stop_words.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())

    # 添加关键词
    my_words = ['陈凯歌', '尔冬升', '赵薇', '郭敬明', '小四', '大鹏', '李诚儒']   
    for i in my_words:
        jieba.add_word(i) 

    # 自定义停用词
    my_stop_words = ['哈哈哈', '哈哈哈哈', '评论']
    stop_words.extend(my_stop_words)               

    # 分词
    word_num = jieba.lcut(x_series.astype('str').str.cat(sep='。'), cut_all=False)

    # 条件筛选
    word_num_selected = [i for i in word_num if i not in stop_words and len(i)>=2]
    
    return word_num_selected
    
text1 = get_cut_words(x_series=df_all[df_all.content.str.contains('凯歌')]['content'])
text1[:5] 

# 绘制词云图
def gen_my_stylecloud(text, file_name, icon_name='fas fa-heart'):
    stylecloud.gen_stylecloud(text=' '.join(text), max_words=1000,
                              collocations=False,
                              font_path=r'‪C:\Windows\Fonts\msyh.ttc',
                              icon_name=icon_name,
                              size=653,
                              output_name=f'../image/{file_name}.png'
                             )   


gen_my_stylecloud(text=text1, file_name='弹幕角色陈凯歌-词云图') 

text2 = get_cut_words(x_series=df_all[df_all.content.str.contains('尔冬升')]['content'])
text2[:5] 
gen_my_stylecloud(text=text2, file_name='弹幕角色尔冬升-词云图', icon_name='fas fa-star')

text3 = get_cut_words(x_series=df_all[df_all.content.str.contains('赵薇')]['content'])
text3[:5] 
gen_my_stylecloud(text=text3, file_name='弹幕角色赵薇-词云图', icon_name='fas fa-comments')


text4 = get_cut_words(x_series=df_all[df_all.content.str.contains('郭敬明|小四')]['content'])
text4[:5] 

gen_my_stylecloud(text=text4, file_name='弹幕角色郭敬明-词云图', icon_name='fas fa-leaf')

text5 = get_cut_words(x_series=df_all[df_all.content.str.contains('李诚儒')]['content'])
text5[:5]  

gen_my_stylecloud(text=text5, file_name='弹幕角色李诚儒-词云图', icon_name='fas fa-thumbs-up') 

text6 = get_cut_words(x_series=df_all[df_all.content.str.contains('曹骏')]['content'])
text6[:5]  

gen_my_stylecloud(text=text6, file_name='弹幕角色曹骏-词云图') 


   








