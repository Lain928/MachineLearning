# 导入库
import pandas as pd 
# 导入数据
df = pd.read_csv('../data/weibo.csv', header=None, names=['name', 'number', 'day'])
df.head() 
df.shape
from datetime import datetime

def transform_day(x): 
    x = '2020年' + x 
    date_format = datetime.strptime(x, '%Y年%m月%d日')
    return datetime.strftime(date_format, '%Y-%m-%d') 
   
   
df['day'] = df.day.apply(transform_day)
df.head() 

# 筛选数据
# df_sel = df[df['day'] >= '2020-10-02']
# df_sel.head() 

df_resuluts = pd.pivot_table(data=df, 
                             index='name', 
                             columns='day', 
                             values='number', 
                             aggfunc='mean', 
                             fill_value=0
                            )
df_resuluts.head() 
df_resuluts.to_csv('../data/df_resuluts.csv') 
