# 자연어 임베딩=피쳐=벡터화
- 벡터의 유사도:단어토큰간의 유사도를 벡터화해서 유사도 측정하기 위함
- 코사인유사도
- 자카드
- 유클리디안 유사도
- 멘하탄 유사도
---
- Ex1 코사인 유사도를 사용하기
1. 영화 대상으로 수집, 유사도를 비교하여 해당 단어(제목)의 속성(리뷰)과 유사한 단어(제목) 5개를 추출한다.
2. 리뷰 기준으로 각 제목당 2만개의 리뷰를 추출
3. 결측치를 제거한다.
4. 벡터화를 한다.
5. 각리뷰하나당 2만개의 리뷰를 비교하여  코사인유사도 이용
6. 키,밸류로 튜플화 (제목,리뷰)
7. 함수 생성 (인덱스,제목)으로 다시 정렬
8. 제목만 꺼내오도록 설정
9. 함수('제목') 으로 1번 실행
```py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity #코사인 유사도
data = pd.read_csv('data1.csv', low_memory=False)
data = data.head(20000)#2만개의 리뷰

#결측치 제거
data['overview'] =data['overview'].fillna('')#오버뷰기준으로 함, 나머진 의미없는 데이터
tfidf = TfidfVectorizer(stop_words='english')#불용성 처리, 벡터처리, 단어가 등장하는 문서도 고려
tfidf_t = tfidf.fit_transform(data['overview'])#실수로 처리
#선입력 데이터별 후 입력 데이터 모두와 코사인 유사도 계산

#각 리뷰의 유사도
cos = cosine_similarity(tfidf_t,tfidf_t)#(2민*2만)코사인유사도, 기울기로 비교,2만곱하기 2만이라 2차원데이터가됨
#vocabulary 화
t_idx =dict(zip(data['title'],data.index)) #키값에 대응되는 밸류 값으로 튜플화(제목,리뷰)
#영화 제목순으로 정렬

#이함수가 바로 영화 추천순으로 정렬된것
def ck_s_t(t,cosine_sim=cos):
    idx=t_idx[t]
    c_sc=list(enumerate(cosine_sim[idx]))
    c_sc=sorted(c_sc, key = lambda x: x[1],reverse=True)#내림차순 정렬,영화 제목순으로 정렬
    m_i =c_sc[1:6]#데이터를 꺼내올때는 자기자신과 비교하지않게 주의,5개만 ㅇㅇ
    m_idx=[i[0] for i in m_i]#제목만 꺼내기
    return data['title'].iloc[m_idx]
#tit_data = t_idx.keys()
#star wars
ck_s_t('Star Wars')#해당 영화의 리뷰와 비슷한 리뷰를 가진 영화 추출
''' 5개 목록
1154    The Empire Strikes Back
1167         Return of the Jedi
1267               Mad Dog Time
5187        The Triumph of Love
309           The Swan Princess

'''


```
1. 자카드 유사도
- 원리 : len(교집합)/len(합집합)
```py
#원리
data1 = '안녕 나는 오늘 힘들어'.split()
data2 = '안녕 못해 나는 지금 너무 힘들어'.split()
un = set(data1) | set(data2)#합집합
intd = set(data1) & set(data2)#교집합
```
- 데이터 벡터화
```py
data1 = '안녕 나는 오늘 힘들어'
data2 = '안녕 못해 나는 지금 너무 힘들어'
#정보통합
from sklearn.feature_extraction.text import TfidfVectorizer
t_v=TfidfVectorizer()#문장 벡터화
m_data= t_v.fit_transform([data1,data2])
```
- 모듈 사용(자카드)

```py
from sklearn.metrics import jaccard_score
import numpy as np
jaccard_score(np.array([0,1,0,0]),np.array([0,1,1,2]),average=None)

```
2. 유클리디안 유사도
- 점과 점의 사이의 거리를 이용함
```py
#원리
def f(A,B):#유클리드거리 연산식,클래스 사용 X
    return np.sqrt(np.sum((A-B)**2))#루트계산
A = np.array([0,1,2,3,4])
B = np.array([1,0,1,2,3])
f(A,B)#A와B사이의 거리가 도출됨

```
- 모듈사용
```py
from sklearn.metrics import euclidean_distances #유클리디안 연산
euclidean_distances(m_data[0:1],m_data[1:2])
#array([[1.0486415]])
def e_f(m):#정규화
    return m/np.sum(m)
e_data = e_f(m_data)
euclidean_distances(e_data[0:1],e_data[1:2])
#array([[0.23884449]])
```
3. 멘하탄 유사도
- 멘하탄 거리를 이용하여 유사도 측정
- 원리
```py 
a=np.array([-1,2,3])
b=np.array([1,3,-4])
abs(a-b)
np.sum(abs(a-b))#멘하탄 연산
#함수로 정의
def f(A,B):
    return np.sum(abs((A-B)))
    A = np.array([0,1,2,3,4])
B = np.array([1,0,1,2,3])
f(A,B)
```
- 모듈 사용
```py
from sklearn.metrics.pairwise import manhattan_distances
manhattan_distances(m_data[0:1],m_data[1:2])#멘하탄,격자를거쳐서 짧은거리 찾음
#array([[2.38220441]])
manhattan_distances(e_data[0:1],e_data[1:2])#정규화된 데이터 간 길이
#(array([[0.37672893]])

ck1=manhattan_distances(m_data[0:1],m_data[1:2])
ck2=manhattan_distances(m_data[1:2],m_data[2:3])
ck1=ck1/ck1
ck2=ck2/ck1

e_ck1=manhattan_distances(m_data[0:1],m_data[1:2])
e_ck2=manhattan_distances(m_data[0:1],m_data[2:3])
e_ck1=e_ck1/e_ck1
e_ck2=e_ck2/e_ck1
print("맨하탄 거리를 이용한 거리 계산")
print(f'정규화 전 data를 이용한 거리 계산 {abs(ck1-ck2)}')
print(f'정규화 후 data를 이용한 거리 계산 {abs(e_ck1-e_ck2)}')
'''
맨하탄 거리를 이용한 거리 계산
정규화 전 data를 이용한 거리 계산 [[1.76734191]]
정규화 후 data를 이용한 거리 계산 [[1.91314919]]
'''
```
---
## Q1
1. data 로드를 통한 데이터 프레임 완성
2. 한글에 맞게 형태소 분석기를 이용하여 단어 토큰화 작업 후 TF-IDF를 계산
3. 코사인 유사도 기반을 통한 영화 3종 추천
```py
#1.데이터로드 후  데이터프레임화
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity #코사인 유사도
data = pd.read_csv('data2.csv')

#2.단어 토큰화 작업 후 TFIDF화
from soynlp import DoublespaceLineCorpus 
from soynlp.word import WordExtractor
from konlpy.tag import Okt
tfidf=TfidfVectorizer()
otk = Okt()
data_l=[' '.join(otk.morphs(i)) for i in data['content']]
제목=list(data['name'])
d_l=[]
for i in data['content']:
    d_l.append(i)
#tf idf 계싼
m_data= tfidf.fit_transform(제목,d_l)
m_data.toarray()

#3.코사인 유사도 사용해서,유사도 기반 3종추출
tfidf = TfidfVectorizer()
tfidf_t = tfidf.fit_transform(d_l)#18의 벡터화된리뷰로 학습

cos = cosine_similarity(tfidf_t)# 18
t_idx =dict(zip(data['name'],data.index))

def ck_s_t(t,cosine_sim=cos):
    idx=t_idx[t]
    c_sc=list(enumerate(cosine_sim[idx]))
    
    c_sc=sorted(c_sc, key = lambda x: x[1],reverse=True)
    
    m_i =c_sc[1:4]
    m_idx=[i[0] for i in m_i]
    return data['name'].iloc[m_idx]
ck_s_t('올드보이')
'''
5     친절한금자씨
11    어바웃 타임
1        노트북
'''
c1=manhattan_distances(tfidf_m,tfidf_m)#숫자가 높으면 유사도는?->낮다
c2=euclidean_distances(tfidf_m,tfidf_m)#
ck_s_t('올드보이',c1)
'''
8       아저씨
7       신세계
5    친절한금자씨
'''
ck_s_t('올드보이',c2)
'''
17        스타워즈
3       니모를찾아서
14    지금만나러갑니다
'''
```
### Q2 다른 유사도 방식 이용
```py
import numpy as np
from sklearn.metrics import euclidean_distances 
euclidean_distances(m_data[0:1],m_data[1:2])

from sklearn.metrics.pairwise import manhattan_distances
manhattan_distances(m_data[0:1],m_data[1:2])

def e_f(m):
    return m/np.sum(m)
e_data = e_f(m_data)

euclidean_distances(e_data[0:1],e_data[1:2])

manhattan_distances(e_data[0:1],e_data[1:2])
가...아니라 
ck_s_t('올드보이',c1)
ck_s_t('올드보이',c2)
```

### 회귀의 문제로 분석
- 머신러닝은 모델의 학습, 모델의 규칙은 상황에 달라진다
- 딥러닝은 모델을 신경망을 이용하여 구축한다
#### 딥러닝으로 회귀문제 해결
```py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

#모델
m=Sequential()#모델생성
m.add(Dense(1,input_dim=1,activation='linear'))#모델생성(모양 결정)
sgd = optimizers.SGD(lr=0.01)#오차수를 어떻게 최적화 할것인지 규칙 설정
m.compile(optimizer=sgd, loss='mse', metrics=['mse'])#모델의 학습 규칙 설정
m.fit(X,Y,epochs=2)# 학습방법 = 지도학습
import matplotlib.pyplot as plt
plt.plot(X,m.predict(X),'b',X,Y,'k.')
```
- 학습이란 가중치의 갱신
- loss = 작을수록 좋다
- 정확도= 클수록 좋다.
- y = wx + b (기계 학습)
- 단일분류
```py
X=[-40,-30,-20,-10,0,10,20,30,40]
Y=[0,0,0,0,1,1,1,1,1]
m1=Sequential()
m1.add(Dense(1,input_dim=1,activation='sigmoid'))#2진분류, 참이나 거짓이나 판단
m1.compile(optimizer=optimizers.SGD(lr=0.01), loss = 'binary_crossentropy'
          , metrics=['binary_accuracy'])
m1.fit(X,Y,epochs=200)
plt.plot(X,m1.predict(X),'b',X,Y,'k.')

```
- 다중 분류
```py
X=[[-40],[-30],[-20],[-10],[0],[10],[20],[30],[40]]
Y=[[1,0,0],[1,0,0],[1,0,0],
   [0,1,0],[0,1,0],[0,1,0],
   [0,0,1],[0,0,1],[0,0,1]]
m2=Sequential()
m2.add(Dense(3,input_dim=1,activation='softmax'))#2진분류, 참이나 거짓이나 판단
m2.compile(optimizer='adam', loss = 'categorical_crossentropy'
          , metrics=['accuracy'])
m2.fit(X,Y,epochs=200)
plt.plot(X,m2.predict(X),'b',X,Y,'k.')
```

### Q3 데이터 로드 분류
- 판다스 패키지로 csv 로드

```py
import pandas as pd
df=pd.read_csv('data3.csv',encoding='latin1')
#데이터 시각화(데이터 분석)
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style = 'ticks',color_codes = True)
g = sns.pairplot(df,hue='Species')

#data 정리 , 분류를 할것이기에 유사도가 높은 데이터는 제외한다
import numpy as np
#'Iris-setosa', 'Iris-versicolor','Iris-virginica'
#Y데이터가 될 Sepecies의 3가지 데이터를 정수화 해야한다.
np.unique(df['Species'])

X_data = df[['SepalLengthCm','PetalLengthCm','PetalWidthCm']].values
#Y_data =df['Species'].values#정수인코딩해야함
Y_data =df['Species'].replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [0,1,2])#원핫,인코딩 말고 ..정수화
Y_data=Y_data.values#우리가 필요한건.. 밸류값

from tensorflow.keras.utils import to_categorical #입력 data 가 반드시 정수 일때만 사용
Y_data = to_categorical(Y_data)
```

1. 데이터 모델 생성 및 학습(입출력층 갯수는 변해선 안됨)

```py
from sklearn.model_selection import train_test_split
t_x,tt_x,t_y,tt_y = train_test_split(X_data,Y_data,random_state=1)

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
m=Sequential()
m.add(Dense(3,input_dim=3,activation='softmax'))#2진분류, 참이나 거짓이나 판단
m.compile(optimizer='adam', loss = 'categorical_crossentropy'
          , metrics=['accuracy'])
hy= m.fit(t_x,t_y,epochs=200,validation_data=(tt_x,tt_y),batch_size=1)

ec=range(1,len(hy.history['accuracy'])+1)
plt.plot(ec,hy.history['loss'])
plt.plot(ec,hy.history['val_loss'])
```
### MLP 다중 퍼셉트론 (입력, 은닉, 출력)
```py
#레이어층하나만 더쌓으면 mlp구조가됨
m=Sequential()
m.add(Dense(3,input_dim=3,activation='sigmoid'))
m.add(Dense(3,activation='softmax'))
m.compile(optimizer='adam', loss = 'categorical_crossentropy'
          , metrics=['accuracy'])
hy= m.fit(t_x,t_y,epochs=200,validation_data=(tt_x,tt_y),batch_size=1)

```