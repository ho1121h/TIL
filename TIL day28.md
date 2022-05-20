# 임베딩
- 임베딩 예
```py
#Embedding(단어수,피처차원)
m=Sequential()
m.add(Embedding(10000,100,input_length=80))
m.add(SimpleRNNCell(64,dropout=0.5,return_sequences=True))
m.add(SimpleRNNCell(64,dropout=0.5))
m.add(Dense(1))
tf.sigmoid()

```

1. data로드 분석후 전처리 진행
2. 분석된 내용을 기반으로 문제를 선택하고 학습 후 결과를 계산
3. data를 3종으로 나누고 최종 test 데이터의 점수가 높게 나오도록 자유롭게 모델을 튜닝하여 결과를 만드시오
```py
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense , Embedding,SimpleRNN,Dropout,LSTM,GRU,Layer,Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer#토큰
from tensorflow.keras.utils import to_categorical#원핫인코딩(정수형 범주)에만
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud,STOPWORDS
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
#데이터 가져오기
data = pd.read_csv('spam.csv',encoding='latin1')
data =data[['v1','v2']]
data
#EDA
st=set(stopwords.words('english'))
p_ck=list(punctuation)
st.update(p_ck)

def 웹문서_처리(t):
    su=BeautifulSoup(t,'html.parser')
    return su.get_text()
def 정규표현_정리(t):
    return re.sub('\[[^]*\]','',t)#replace 와 비슷한 정규식임
def 불필요_정리(t):
    return re.sub(r'http\S+','',t)
def 불용어_정리(t):
    f_l=[]
    for i in t.split():
        if i.strip().lower() not in st and i.strip().lower().isalpha():
            f_l.append(i.strip().lower())
    return ' '.join(f_l)
def 모두_정리(t):
    t=웹문서_처리(t)
    t=불필요_정리(t)
    t=불용어_정리(t)
    return t
data['v2']=data['v2'].apply(모두_정리)

data.v1.replace('ham',1,inplace=True)#label 값을 0,1로 변경
data.v1.replace('spam',0,inplace=True)

#워드클라우드로 확인
wc=WordCloud(max_words=2500).generate(' '.join(data[data.v1==1].v2))
plt.imshow(wc)
wc=WordCloud(max_words=2500).generate(' '.join(data[data.v1==0].v2))
plt.imshow(wc)
data['v1'].value_counts().plot(kind='bar')#햄이 더많네
#햄이 더많으면 
t_l_0=data[data.v1==0]['v2'].str.len()#스팸
t_l_1=data[data.v1==1]['v2'].str.len()#햄
fig,(ax1,ax2)=plt.subplots(1,2)
ax1.hist(t_l_0)
ax2.hist(t_l_1)
plt.show()#흠...
t_l_1=data[data.v1==0]['v2'].str.split().map(lambda x: len(x))
t_l_1=data[data.v1==1]['v2'].str.split().map(lambda x: len(x))
fig,(ax1,ax2)=plt.subplots(1,2)
ax1.hist(t_l_0)
ax2.hist(t_l_1)
plt.show()#흠...햄쪽 길이기 길다
import seaborn as sns#히스토그램사용
fig,(ax1,ax2)=plt.subplots(1,2)

t_l_0=data[data.v1==0]['v2'].str.split().apply(lambda x: 
                                                        [len(i) for i in x])
sns.distplot(t_l_0.map(lambda x : np.mean(x)),ax=ax1)
t_l_1=data[data.v1==1]['v2'].str.split().apply(lambda x: 
                                                        [len(i) for i in x])
sns.distplot(t_l_1.map(lambda x : np.mean(x)),ax=ax2)
#모델생성
X_data=data.v2
Y_data=data.v1

tk= Tokenizer()
tk.fit_on_texts(X_data)#토큰학습
tk.index_word
len(tk.word_index)#6029
from tensorflow.keras.preprocessing.sequence import pad_sequences
ck_data= tk.texts_to_sequences(X_data)#토큰 시퀀스화
X_data =pad_sequences(ck_data)#패딩화

Y_data=to_categorical(Y_data)
Y_data


t_x,tt_x,t_y,tt_y = train_test_split(X_data,Y_data,test_size=0.3,random_state= 1)
```
- 모델생성

```py
#1번모델
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es=EarlyStopping(monitor='val_loss',mode='min',verbose=2,patience=4)
mc=ModelCheckpoint()

m = Sequential()#임베딩은 차원을 꼭 맞춰줘야한다
m.add(Embedding(6030,64))#6030을 64 차원으로 축소(관련성 높게)
m.add(SimpleRNN(64,dropout=0.5,return_sequences=True))
m.add(SimpleRNN(32,dropout=0.5))#뉴런 반절을 없애지만 속도는 느려짐
m.add(Dense(2,activation='sigmoid'))
m.compile(optimizer='adam',
         loss='binary_crossentropy',
         metrics=['acc']
         )
hy= m.fit(t_x,t_y,epochs=100,batch_size=64,validation_data=(tt_x,tt_y),callbacks=[es])

#검증
ec=range(1,len(hy.history['acc'])+1)
plt.plot(ec,hy.history['loss'])
plt.plot(ec,hy.history['val_loss'])

#2번모델-LSTM
m2 = Sequential()
m2.add(Embedding(6030,64))
m2.add(LSTM(64,return_sequences=True))
m2.add(LSTM(32,dropout=0.5))
m2.add(Dense(2,activation='sigmoid'))
m2.compile(optimizer='adam',
         loss='binary_crossentropy',
         metrics=['acc']
         )
hy= m2.fit(t_x,t_y,epochs=100,batch_size=64,validation_data=(tt_x,tt_y),callbacks=[es])
ec=range(1,len(hy.history['acc'])+1)
plt.plot(ec,hy.history['loss'])
plt.plot(ec,hy.history['val_loss'])

#3번모델 GRU
m3 = Sequential()
m3.add(Embedding(6030,64))
m3.add(GRU(64,dropout=0.5,return_sequences=True))
m3.add(GRU(32,dropout=0.5))
m3.add(Dense(2,activation='sigmoid'))
m3.compile(optimizer='adam',
         loss='binary_crossentropy',
         metrics=['acc']
         )
hy= m3.fit(t_x,t_y,epochs=100,batch_size=64,validation_data=(tt_x,tt_y),callbacks=[es])
ec=range(1,len(hy.history['acc'])+1)
plt.plot(ec,hy.history['loss'])
plt.plot(ec,hy.history['val_loss'])


#4번모델-LSTM 양방향
m4 = Sequential()
m4.add(Embedding(6030,64))
m4.add(Bidirectional(LSTM(64,dropout=0.5,return_sequences=True)))
m4.add(Bidirectional(LSTM(32,dropout=0.5)))
m4.add(Dense(2,activation='sigmoid'))
m4.compile(optimizer='adam',
         loss='binary_crossentropy',
         metrics=['acc']
         )
hy= m4.fit(t_x,t_y,epochs=100,validation_data=(tt_x,tt_y),callbacks=[es],
          batch_size=64)
ec=range(1,len(hy.history['acc'])+1)
plt.plot(ec,hy.history['loss'])
plt.plot(ec,hy.history['val_loss'])
```
- 검증 점수를 보면 GRU모델이 가장 좋게 나왔음
- 복합성이 높으면 LSTM, 적으면 GRU
---
- 임베링딩동
    - 사람이 사용하는 언어를 컴퓨터가 이해할수있는 숫자 형태인 벡터로 변환한 결과 혹은 일련의 과정을 의미,피쳐의 갯수,차원의 갯수에 의해 공간이 달라짐
    - 원본이 줄여지고 요약된 데이터가된다.
    - idf 에서 빈도가 0 이면 스무딩이 일어남
- 예측 기반 임베딩
    - 특정 문맥에서 어떤 단어가 나올지 예측
    - 대표적으로 워드 투 벡터가 있다.
- 워드 투 벡터
    - 신경망 알고리즘으로 주어진 텍스트에서 텍스트의 각 단어마다 한씩 일련의 벡터를 출력
    - 이때 서로 가깝다는 의미는 코사인 유사도를 이용하여 단어 간의 거리를 측정한 결과로 나타나는 관계성을 의미



# 예측기반 임베딩,워드 투 벡터
```py
from nltk.tokenize import sent_tokenize , word_tokenize #단어 쪼개는도구
import gensim
from gensim.models import Word2Vec ,FastText
s_data = open('data1.txt','r',encoding='utf-8').read()
s_data = s_data.replace('\n',' ')
d_l = []
for 문장 in sent_tokenize(s_data):
    s_l=[]
    for 단어 in word_tokenize(문장):
        s_l.append(단어.lower())
    d_l.append(s_l)
d_l#이대론 연관성이 없다.벡터를 제시하되 연관성과 함께

#Word2Vec (입력 data, 단어 빈도 수,임베딩된 차원, 단어갯수, sg=0=CBOW,1=sikp_gram))
#하나의 모델이 완성
w_m1= Word2Vec(d_l,min_count=1,vector_size=100,window=5)
w_m1.wv.similarity('peter','wendy')#0.074393846
w_m1.wv.similarity('peter','hook')#0.027709913

w_m2= Word2Vec(d_l,min_count=1,vector_size=100,window=5,sg=1)#스킵그램
w_m2.wv.similarity('peter','wendy'),w_m2.wv.similarity('peter','hook')
#(0.40088683, 0.52016735)


```
- 데이터에 따라 성능이 달라짐
- 입력된 데이터에 한정되는 워드투백터의 단점을 보완하고자/학습된 데이터가 아니면 날려버림
- 패스트 텍스트 적용/사전에 없는 단어에 벡터 값을 부여하는 방법
- 패스트텍스트는 주어진 문서의 각 단어를 엔그램으로 표현함
    - 연관 관계성을 특정 토큰이 가짐
```py
from gensim.test.utils import common_texts
#패스트 텍스트
m2 = FastText('data1.txt', vector_size=4,window=3,min_count=1)
m2.wv.similarity('peter','wendy')#0.45924556
m2.wv.similarity('peter','hook')#0.043825187
```

---
- 복습...
통계적언어모델은 조건부확률에 의거하여 
(유니그램,바이그램 ,엔그램)로..
- 단어의 표현
    - 국소표현 해당 단어만 보고 특정값을 맵핑하여 단어를 표현(**tf-idf**,bow,dtm)
    - 분산표현 그단어를 표현하고자 주변을 참고하여 단어를 표현(**워드 투벡터 패스트텍스트**)
- bow 출현 빈도에만 집중
- 문서단어 행렬의 한계: 희소표현(원핫), 단순 빈도수 기반 접근, 통계기반이라서
- 벡터의 유사도: 벡터끼리 공간상 유사하면 회귀문제 생성문제 등을 풀 수 있어서
- 코사인,유클리디안, 자카드,멘하탄 등
- EDA 탐색적 데이터 분석 데이터마다 달라지므로 마음대로 하면됨, 자연어는 큰틀이 존재함. 라벨 분류, 오차확인, 

- 성능 최적화 -데이터를 사용한 최적화, 최대한 많은 데이터 수집, 알고리즘을 이용한 성능 최적화, 임베딩,..lstm 등
- 진단: 성능 향상이 어느 순간 멈추었다면 원인을 분석할 필요가 있다 문제를 진단하는데 사용할수있는것이 모델에 대한 평가
    - 가중치: 가중치에 대한 초기값은 작은 난수를 사용
    - 
- 활성화 함수: 의미전달으 ㅣ목적과 필터의 역할
- 배치와 에포크: 한번에 처리하는 양을 늘릴 수 있으나 메모리 한계때문에..
- 옵티마이저: 갱신하는 방법 주로 아담과 sgd 사용
- 앙상블: 분석의 목적과 통합의 목적. 
- 하이퍼 파라미터 조정
- 배치 정규화 
- RNN
- 복합성이 높으면 LSTM, 복합성이 낮으면 GRU