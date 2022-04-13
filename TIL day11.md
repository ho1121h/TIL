## 웹 크롤링 및 데이터 시각화
```python
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Gulim'
matplotlib.rcParams['axes.unicode_minus']=False
import csv
import pandas as pd
from pandas import DataFrame
import numpy as np

df=pd.read_excel('data1 (2).xlsx')
df



```

---
## 원하는 주식의 정보를 크롤링

```python
#1초기설정

import warnings
warnings.filterwarnings('ignore')
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

#2데이터수집
b = webdriver.Chrome()
url = "http://www.naver.com"
b.get(url)
b.implicitly_wait(10)
b.find_element_by_xpath('//*[@id="NM_FAVORITE"]/div[1]/ul[2]/li[3]/a').click()
b.implicitly_wait(10)
b.find_element_by_xpath('//*[@id="stock_items"]').send_keys("삼성전자\n")
b.implicitly_wait(10)
b.find_element_by_xpath('//*[@id="content"]/div[4]/table/tbody/tr[1]/td[1]/a').click()
b.execute_script("window.scrollTo(0,500)")
b.implicitly_wait(10)
b.find_element_by_xpath('//*[@id="content"]/ul/li[2]/a').click()
b.implicitly_wait(10)
frame = b.find_element_by_xpath('//*[@id="content"]/div[2]/iframe[2]')
b.switch_to.frame(frame)
def f(x): return x.text.strip().replace("\n", "").replace("\t", "").replace(",", "")
l_data = []
for i in range(2, 8):
    if i != 3:
        html = b.page_source
        s = BeautifulSoup(html, 'html.parser')
        b.find_element_by_xpath(f'/html/body/table[2]/tbody/tr/td[{i}]/a').click()
        b.implicitly_wait(10)
        data = s.select('body tr')
        for i in data:
            if i.select_one('span.tah'):
                l_data.append(list(map(f, i.select('span.tah'))))
b.quit()

#3데이터 저장
l_data.reverse()
df = pd.DataFrame(l_data, columns=['날짜', '종가', '전일비', '시가', '고가', '저가', '거래량'])
df.to_excel('Q2.xlsx', index=False)

#4데이터 로드
df=pd.read_excel('Q2.xlsx')
df

#5 종가 시가 저가 고가 출력
plt.plot(df.날짜,df.종가)
plt.title('삼성전자_종가')
plt.xticks(rotation=90)
plt.show()
plt.plot(df.날짜,df.시가)
plt.title('삼성전자_시가')
plt.xticks(rotation=90)
plt.show()
plt.plot(df.날짜,df.저가)
plt.title('삼성전자_저가')
plt.xticks(rotation=90)
plt.show()
plt.plot(df.날짜,df.고가)
plt.title('삼성전자_고가')
plt.xticks(rotation=90)
plt.show()
plt.plot(df.날짜,df.종가,label='삼성전자_종가')
plt.plot(df.날짜,df.시가,label='삼성전자_시가')
plt.plot(df.날짜,df.저가,label='삼성전자_저가')
plt.plot(df.날짜,df.고가,label='삼성전자_고가')
plt.title('삼성전자')
plt.xticks(rotation=90)
plt.legend()
plt.show()
#한문단에 전부 입력해서 출력하면 한페이지에 모든 그래프 출력




```

