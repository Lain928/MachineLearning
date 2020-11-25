import numpy as np
import pandas as pd
df_news = pd.read_csv('Resources/MVI_20011.txt',header = None)
print(df_news)
print(type(df_news))
df_news = df_news.head(14)
df_news.to_csv("Resources/MVI_20011.csv",index=False,header = None)