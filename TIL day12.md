## 데이터 시각화 마지막
-시몬(seaborn)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt#맷플롯 사용 #그래프로 나타냄
import seaborn as sns #시몬 모듈 사용

fmri = sns.load_dataset('fmri')
sns.set_style('whitegrid')
sns.lineplot(x='timepoint',y='signal',data=fmri)
#대충 그래프
fmri.sample(n=10,random_state=1)#샘플 10개를 랜덤으로 가져옴
#대충 표
t=sns.load_dataset('tips')
sns.regplot(x='total_bill',y='tip',data=t)
#대충 그래프 및 산포도
sns.scatterplot(x='total_bill',y='tip',hue='time',data=t)
#이것도 산포도
t=sns.load_dataset('tips')
sns.countplot(x='smoker',hue='time',data=t)#x를 바탕으로 hue를 기준으로
#대충 런치와 디너 기준으로 나눈 표
sns.barplot(x='day',y='total_bill',data=t)#요일 기준으로 막대그래프로  표현
sns.violinplot(x='day',y='total_bill',data=t,hue='smoker')#분포를 나타내는 플롯
sns.swarmplot(x='day',y='total_bill',data=t,hue='smoker')#분포를 나타내는 플롯 2
g = sns.FacetGrid(t,col='time',row='sex')#col 과 row 만큼 그래프 생성
g = sns.FacetGrid(t,col='time',row='sex')
g.map(sns.scatterplot,'total_bill','tip')#맵으로 함수를 각각 적용 
g= sns.FacetGrid(t,col='time',row='sex')
g.map_dataframe(sns.histplot,x='total_bill')
```
## 시몬 사용 2

```python 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rc('font', family='Gulim')
matplotlib.rc('axes', unicode_minus=False)
# 가져올 샘플
붓꽃data=sns.load_dataset('iris')
타이타닉data=sns.load_dataset('titanic')
팁data=sns.load_dataset('tips')
운송data=sns.load_dataset('flights')
#붓꽃을 사용

x=붓꽃data.petal_length.values
sns.rugplot(x)
plt.title("붓꽃data중 꽃잎 길이 러그 플롯")
plt.show()
#러그플롯

sns.kdeplot(x)#x축의 정보량이 얼마나 있는가 ? 
plt.title("붗꽃 data 중 꽃잎 길이 커널 밀도 플롯")
plt.show()
#밀도 플롯
sns.displot(x,kde=True,rug=True)#   밀도  분포
plt.show()
#위 그래프에 막대그래프 추가 러그 추가
sns.jointplot(x='sepal_length',y='sepal_width',data=붓꽃data, kind='kde')#시각화 등고선
sns.pairplot(붓꽃data)#플롯에 맞게 히스토 그램 각 데이터를 모두 표현
sns.pairplot(붓꽃data,hue='species')
plt.title("종으로 시각화")#종 별로 색입힘
plt.show()

## 타이타닉 사용###
sns.countplot(x='class', data=타이타닉data)
# 클래스 기준으로 막대그래프

타이타닉_sub_data=타이타닉data.pivot_table(index='class',columns='sex',aggfunc='size')
타이타닉_sub_data#클래스와 성별 뽑아옴

sns.heatmap(타이타닉_sub_data,cmap=sns.light_palette('gray',as_cmap=True),
            annot=True,fmt='d')
#가장많은 영향도를 가졌으면 진함

##운송데이터 사용##

data=운송data.pivot('month','year','passengers')
data
sns.heatmap(data, annot=True, fmt='d',linewidths=1)#히트맵으로 나타냄, 수치를 나타냄, 정수로 나타냄
```
## konlpy 사용
- 의미 있는 텍스트를 추출하여 얼마나 분포되어 있는가 확인가능
- word cloud 문서으 ㅣ키워드 개념등 직관적으로 파악하게 핵심단어를 시각화 함
    시각적인 중요도를 강조를 위해
- 빅데이터를 분석할때 특징을 확인하기 위해

```python
from konlpy.tag import *
ok=Okt()
print(ok.nouns('안녕 나는 파이썬을 열심히 공부 하고 있어. 너는 어때?'))
from wordcloud import WordCloud ,STOPWORDS
import matplotlib.pyplot as plt
폰트='C:\Windows\Fonts\malgun.ttf' #폰트 설정
'''
WordCloud(max_words="최대 수용 잔어 갯수",background_color='배경색상',
          font_path='폰트설정(사용할 파일의 경로)',
          width='넓이',
          height='높이값'
'''
wc=WordCloud(font_path=폰트,background_color='white')
text='파이썬 코로나 파이썬 삼성 멀티캠 워드 워드클라우드 워드 삼성 나는 나 이런 이상 파이썬 코로나 데이터 자료 파이썬'
wc=wc.generate_from_text(text)
plt.figure(figsize=(10,10))
plt.imshow(wc,interpolation='lanczos')
plt.axis('off')
plt.show()
####
st_w=set(STOPWORDS)#위 워드클라우드 셋
st_w.add('파이썬')#파이썬을 추가
wc=WordCloud(font_path=폰트,background_color='white',stopwords=st_w).generate(text)
plt.figure(figsize=(10,10))
plt.imshow(wc,interpolation='lanczos')
plt.axis('off')
plt.show()
####
text2='나는 파이썬 을 공부 중 입니다. 새로운 문장 을 만드 는 공부 중 입니다. 공부'
ck={'파이썬':3,'공부':1,'나는':2}
wc2= WordCloud(font_path=폰트)
wc2=wc2.generate_from_text(text2)#횟수만 고려해서 강조를 결정
wc2=wc2.generate_from_frequencies(ck)#문자의 우선순위를 고려하여 표현
plt.figure(figsize=(10,10))
plt.imshow(wc2)
plt.axis('off')
plt.show()
###
from collections import Counter
l=['안녕','안녕','안녕','안녕','안녕','나는','나는','나는','나는','파이썬','파이썬','파이썬']
c=Counter(l)
wc3= WordCloud(font_path=폰트)
g_data=wc3.generate_from_frequencies(c)
plt.figure()
plt.imshow(g_data)
plt.axis('off')
plt.show()
plt.savefig('데이터 이름.png')

```
## 웹 크롤링 후 워드클라우드 사용
```python
from wordcloud import WordCloud #워드클라우드모듈
import matplotlib.pyplot as plt #그래프모듈
from collections import Counter#횟수적용모듈
from konlpy.tag import Okt
1.불러오기
with open('가져올 파일','r',encoding='utf-8') as f:
    text=f.read()
text
2.텍스트 정렬
ok_t=Okt()
data1=ok_t.nouns(text)
data1
3.좀더 의미있게
data2=[i for i in data1 if len(i)>1]
data2
4.텍스트 숫자 매김
data3=Counter(data2)
data3
5. 시각화
wc=WordCloud(font_path="C:\Windows\Fonts\malgun.ttf",max_words=100,max_font_size=250)
g_data=wc.generate_from_frequencies(data3)
plt.figure()
plt.imshow(g_data)
plt.axis('off')
plt.show
#가장 많은 단어가 제일 큰 글자로 나온다.
```