## pandas
- 판다스를 이용해 csv 나 엑셀 텍스트 파일을 가져와보자
- 파일을 읽어올때는 인코딩이 필요하다.
```python
import pandas as pd
from pandas import DataFrame
df_data=pd.read_csv('경찰청 강원도경찰청_음주교통사고 발생 현황_20201231.csv',encoding='euc-kr')
df=DataFrame(df_data)

df1=DataFrame(df,columns=["연도","발생"])#컬럼중 연도와 발생 부분만 출력
df1
'''
    연도	발생
0	2020	620
1	2019	493
2	2018	680
3	2017	780
4	2016	708
'''

df2=df.set_index('연도') #연도 기준으로 셋해서 출력
df2
'''
	   발생	사망 부상
연도			
2020	620	11	1053
2019	493	18	797
2018	680	14	1165
2017	780	18	1338
2016	708	18	1266
인덱싱 넘버가 연도로 치환되서 출력됨을 볼 수 있다.

'''
#iloc,loc
df.iloc[:2,:2]#축별 내용을 인덱스로 취급하여 동작 가능하도록 설정하는 함수
'''

    연도	발생
0	2020	620
1	2019	493

0~1 인덱싱을 슬라이싱해서 출력됨을 볼 수 있다. 표 기준 세로가 고차원
'''
df.loc[0]#인덱스를 통한 data 호출이 가능하도록 설정하는 함수(모든 data가 적용될 수있다,)
df.index=df['연도']#인덱스 추가
del df['연도']#해당 인덱스 삭제
df.head()#헤드로 재정렬 오름차순
df.loc[2020]#2020 데이터 호출
n_df=df.reset_index()
print(list(n_df.loc[0]))
n_df.drop(1)#
'''

[2020, 620, 11, 1053] n_df의 0인덱스 부분 출력 몰론 고차원 기준

    연도	발생 사망 부상
0	2020	620	11	1053
2	2018	680	14	1165
3	2017	780	18	1338
4	2016	708	18	1266

drop(1)으로 인해 1인덱스가 삭제됨을 볼 수있다.


'''
```
---
## 데이터 추출 후 저장 및 데이터 프레임 출력
금융 데이터를 수집하여 data테이블을 완성하시오
1.동적크롤링을 이용하여 data를 수집 해주세요
2.정해진 data를 스크레핑 하고 .csv파일로 저장하시오 
3.판다스를 이용하여 data테이블을 구성하시오
4.등락률을 기준으로 + - 에따른 2개의 DF를 만들어주세요

- 초기 
```python
import warnings
warnings.filterwarnings('ignore')
import csv
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd 
#불러오기,추출
b=webdriver.Chrome()
b.implicitly_wait(10)
b.get("http://naver.com")
b.find_element_by_xpath('//*[@id="query"]').send_keys("금융\n")
b.implicitly_wait(10)
b.find_element_by_xpath('//*[@id="web_1"]/div/div[2]/div[2]/a').click()
b.implicitly_wait(10)
b.switch_to.window(b.window_handles[1])#html 안에 html이 존재하기 때문에 스위치를씀
b.find_element_by_xpath('//*[@id="menu"]/ul/li[4]/a/span').click()
b.implicitly_wait(10)
b.find_element_by_xpath('//*[@id="tab_section"]/ul/li[2]/a/span').click()
b.implicitly_wait(10)
b.switch_to.frame('frame_ex2')#스위치
def f(x): return x.text.strip().replace("\n", "").replace("\t", "")#리턴값에 불필요한 값 삭제
title=''
l = []#빈 공간
for page in range(1,5):#페이지 설정
    b.find_element_by_xpath(f'/html/body/div/div/a[{page}]').click()
    print(f"{page}페이지 진행 중")
    html = b.page_source
    s=BeautifulSoup(html, 'html.parser')
    data = s.select_one('tbody').select('tr')
    title = s.select_one('thead').select("th")
    for i in data:
        l.append(map(f, i.select('td')))#map 매서드로 td 태그 전부 추가
b.quit()
#2.저장
with open('data.csv','w',encoding='utf-8-sig',newline='') as f_csv:
    writer=csv.writer(f_csv)
    writer.writerow(map(f,title)) #1차원 데이터 설정
    writer.writerows(l) #2차원 데이터로 저장
#3
df=pd.read_csv('data.csv')
df
'''
	통화명  	        심볼	현재가	전일대비	등락율
0	영국 파운드/달러	GBPUSD	1.3067	0.0021	- 0.16%
1	유로/달러	        EURUSD	1.0910	0.0010	- 0.09%
2	호주 달러/달러	    AUDUSD	0.7474	0.0061	- 0.80%
3	달러/홍콩 달러	    USDHKD	7.8375	0.0009	+0.01%
4	달러/일본 엔	    USDJPY	123.9100	0.1800	+0.14%
...	...	...	...	...	...
107	달러/페루 뉴솔	    USDPEN	3.7200	0.0210	+0.56%
108	달러/폴란드 즈롤티	USDPLN	4.2512	0.0096	- 0.22%
109	달러/프랑스퍼시픽 프랑	USDXPF	109.7500	0.1000	- 0.09%
110	달러/필리핀 페소	USDPHP	51.4230	0.0320	+0.06%
111	달러/호주 달러	    USDAUD	1.3378	0.0109	+0.82%


'''
#4
t1=df[(df['등락율']>'-') & (df['등락율']!='0.00%')].reset_index(drop='index')
print(t1)
t2=df[df['등락율']<'-'].reset_index(drop='index')
print(t2)

c=t1.loc[0,'심볼']
c1=t1.iloc[0,1]
c1
#>> 'GBPUSD'
c=t1['심볼']#t1의 해당 인덱스 가져옴
c
'''
0     GBPUSD
1     EURUSD
2     AUDUSD
3     USDILS
4     USDIRR
5     USDGEL

............
23    USDKMF
24    USDTHB
25    USDPYG
26    USDPLN
27    USDXPF

'''
c_t=t1.set_index(t1['심볼'])#심볼을 0인덱스로 땡겨옴
d=c_t.loc['GBPUSD','통화명']
#->영국 파운드/달러 출력됨
d=c_t.iloc[:2,:2]#슬라이싱
'''
            0인덱       1인덱
	        통화명	        심볼
심볼		
GBPUSD	영국 파운드/달러	GBPUSD          -0인덱
EURUSD	유로/달러	       EURUSD          -1인덱
'''
ck=pd.read_csv('data.csv',skiprows=1)
ck
'''
	영국 파운드/달러	GBPUSD	1.3067	0.0021	- 0.16%
0	유로/달러	EURUSD	1.0910	0.0010	- 0.09%
1	호주 달러/달러	AUDUSD	0.7474	0.0061	- 0.80%
2	달러/홍콩 달러	USDHKD	7.8375	0.0009	+0.01%
3	달러/일본 엔	USDJPY	123.9100	0.1800	+0.14%
4	달러/캐나다 달러	USDCAD	1.2585	0.0082	+0.65%

~~스킵~~
통화명 심볼 현재가 전일대비 등락율 (타이틀)
이 스킵으로 인해 사라짐을 볼 수 있다.

'''

import numpy as np
np.array([[[[1,2,3,4,5]]]]).shape# 그차원 의 원소가 몇개인지 가르켜줌 4차원1개 3차원 1개 2차원1개 1차원 5개
```
