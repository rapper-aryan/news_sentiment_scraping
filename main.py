from bs4 import BeautifulSoup
import pandas as pd
import requests
from textblob import TextBlob
import numpy as np


#Sentiment Function
def sentiments(text):
    blob=TextBlob(text)
    return blob.sentiment.polarity


news_heading=[]
news_desc=[]
source=[]
time=[]
html_text = requests.get('https://www.lemonde.fr/en/international/').text
soup = BeautifulSoup(html_text, 'lxml')
news = soup.find_all('div', class_='thread')
for new in news:
    news_heading.append(new.find('h3', class_='teaser__title').text)
    time.append(new.find('span',class_='meta__date').text)
    news_desc.append(new.find('p', class_='teaser__desc').text)
    source.append(new.find('a'))


df=pd.DataFrame(list(zip(news_heading,news_desc,time,source)),columns=['News Heading','News Description','Date','Source'])
df['sentiment']=df['News Heading'].apply(sentiments)
df['Sentiment Class']=np.where(df['sentiment']<0,"negative",
                               np.where(df['sentiment']>0,'positive','neutral'))

df.to_csv('news_data.csv',index=False)
print(df)

