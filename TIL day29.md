# 웹크롤링 복습
1. 스팀에서 판매순위를 크롤링해보자

```py
import time
import requests
import csv
from selenium import webdriver
from bs4 import BeautifulSoup

b=webdriver.Chrome()

b.maximize_window()
url="https://store.steampowered.com/search/?filter=topsellers"
b.get(url)
b.implicitly_wait(10)
f=open("ex2.csv","w",encoding='utf-8-sig',newline="")
writer=csv.writer(f)
content="최고 매출","평가","가격"
writer.writerow(content)
in_data=[]
for i in range(1,2) :#모든 내용 로드를 위한 동작
    info_n = b.execute_script("return document.body.scrollHeight")#로드된 내용의 최하단 크기확인
    b.execute_script("window.scrollTo(0, document.body.scrollHeight)")#로드된 내용의 최하단으로 이동
    # 확장
    time.sleep(2)
    next_n = b.execute_script("return document.body.scrollHeight")
    if info_n == next_n:
        break
    s=BeautifulSoup(b.page_source,'html.parser')
    data = s.select('div.responsive_search_name_combined')
    
    for i in data:
        if i.div:   
            in_data.append([i.span.text.strip(),
                           i.find_all('div',attrs ={'class':'col search_reviewscore responsive_secondrow'}),
                           i.div.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.text.strip()])

    time.sleep(2)
    
    
writer.writerows(in_data)
f.close()
#html 태그라서 평가를 가져올 수 가 없다. => 방법 모름

import pandas as pd
import numpy as np
data =pd.read_csv("ex2.csv",encoding='utf-8-sig')[['최고 매출','평가','가격']]
data
#html을 통째로 가져와 버렷다
```
```py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
from nltk.corpus import stopwords
import re
data['평가'] = data['평가'].str.replace("[^0-9 ]","")
data['가격'] = data['가격'].str.replace("\n","")
data['가격'] = data['가격'].str.replace('^ +', "")#문장 앞의 공백 제거

data

```
### 문제: 크롤링은 됐는데 html문서를 전부 가져와버림..
- fixed 크롤링

```py
import time
import requests
import csv
from selenium import webdriver
from bs4 import BeautifulSoup
import os
import sys
from selenium.webdriver.common.keys import Keys
import chromedriver_autoinstaller
import warnings
warnings.filterwarnings('ignore')

options = webdriver.ChromeOptions()
options.add_argument('window-size=1280,800')

b = webdriver.Chrome('chromedriver', options=options)
b.implicitly_wait(5)

b.get(url='https://store.steampowered.com/search/?filter=topsellers')
b.implicitly_wait(10)

평가_list = []
title_list = []
for i in range(1,2) :#모든 내용 로드를 위한 동작
    info_n = b.execute_script("return document.body.scrollHeight")#로드된 내용의 최하단 크기확인
    b.execute_script("window.scrollTo(0, document.body.scrollHeight)")#로드된 내용의 최하단으로 이동
    # 확장
    time.sleep(2)
    next_n = b.execute_script("return document.body.scrollHeight")
    if info_n == next_n:
        break
    
    
    article_raw = b.find_elements_by_class_name('responsive_search_name_combined')
    # 제목 크롤링 시작
    for article in article_raw:
        title = article.find_element_by_class_name('title').text# 태그면 겟 에트리뷰트로 가져와야함
        title_list.append(title)
        try:
            평가 =article.find_element_by_css_selector("span.search_review_summary").get_attribute("data-tooltip-html")
            평가_list.append(평가)
        except:
            평가_list.append("")
    
 #  ''' for article in article_raw:
  #      평가= article.find_element_by_class_name('search_review_summary').get_attribute("data-tooltip-html")
   #     평가_list.append(평가)'''
        
    '''for article in article_raw:
        평가 = article.find_element_by_class_name('search_review_summary').
        #평가 = article.get_attribute("data-tooltip-html")#html의 태그를 가져옴,텍스트로..
        평가_list.append(평가)'''
    time.sleep(1) 
import pandas as pd
import numpy as np
# 수집된 url_list, title_list로 판다스 데이터프레임 만들기
df = pd.DataFrame({'매출 순위':title_list,'평가':평가_list})
df

df.to_csv("steam.csv", encoding='utf-8-sig')
```
```
for article in article_raw:
        평가 = article.find_element_by_class_name('search_review_summary').
        #평가 = article.get_attribute("data-tooltip-html")#html의 태그를 가져옴,텍스트로..
        평가_list.append(평가)

        를

        평가 =article.find_element_by_css_selector("span.search_review_summary").get_attribute("data-tooltip-html")
            평가_list.append(평가)로 고치니 됨```