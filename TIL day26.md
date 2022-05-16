# EDA (탐색적 데이터 분석)
- 데이터 전처리 후 학습 했을때의 결과가 이상할때(생각보다 성능이 안나올때) 다시 되돌아봄
- 하나씩 분석하면서 확인해야함.
- 데이터가 표현하는 현상을 더 잘이해하고, 데이터에 대한 잠재적인 문제를 발견 가능

```py
#데이터 로드
import pandas as pd
t_d = pd.read_csv('data1 (1).csv')
ck_data = t_d.review#리뷰데이터가 피쳐이니 피쳐 데이터 정의
#토큰화 클래스없이 토큰화
for i in ck_data:
    print(i.split())
    break#각문장을 리스트로 담고 있음을 확인 가능

t_l=[w.split() for w in ck_data]#문자열 문장 리스트를 토큰화
n_t_l=[len(w) for w in t_l]#토큰화된 리스트에 대한 각 길이 저장
r_n_t_l= [len(w.replace(' ','')) for w in ck_data]#음절의 길이를 저장
max(r_n_t_l)#11235
#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))#그래프에대한 이미지크기
plt.hist(n_t_l,alpha=0.5,bins=50,label='n',color='r')#단어 갯수에대한 히스토그램
plt.hist(r_n_t_l,alpha=0.5,bins=50,label='r_n',color='b')#알파벳 길이에대한 히스토그램
plt.yscale('log', nonposy='clip')
plt.legend()
# 패딩하기엔 너무길고.. 0이 너무 많이 들어가게 된다.(제로패딩)
#보통 저런걸 노이즈라고 부른다. (그래프상 비어있는 곳)

np.max(n_t_l),np.min(n_t_l),np.mean(n_t_l),np.std(n_t_l)
#문장최대길이 2470,문장최소길이10, 문장 평균길이 233.79, 문장길이 표준편차:173.73
#박스플롯(문장의 알파벳 개수를 나타내는 박스 플롯)
plt.boxplot([n_t_l],showmeans=True)#이상치가 심한 데이터 확인: 이상치가 심하면 학습이 잘이루어 지지 않는다.

t_d= pd.read_csv('data2 (1).csv')

x_ck=t_d['review']
x_ck
y_ck=t_d['sentiment']
y_ck
from wordcloud import WordCloud
c=WordCloud().generate(' '.join(t_d['review']))
plt.imshow(c)#br이란 쓸때 없는 단어분포가 많다.html태그임
t_d.describe()   #데이터 프레임
'''
	sentiment
count	25000.00000
mean	0.50000
std	0.50001
min	0.00000
25%	0.00000
50%	0.50000
75%	1.00000
max	1.00000
'''
ck_df=t_d[['review','sentiment']]
ck_df
#분류문제이면 분포를 중요하게 봐야한다.
import seaborn as sns
sns.countplot(ck_df.sentiment)#긍정 부정 분포확인
#1:1 비율이라 데이터의 균형이 아주좋다
plt.hist(n_t,bins=50)
plt.yscale('log',nonposy='clip')#로그 그래프로 보기
ck_df.isna().sum()#결측치 이상무


t_d= pd.read_csv('data2 (1).csv')
t_d=t_d[['review','sentiment']]
t_d
n_t=t_d['review'].apply(lambda x : len(x.split(' ')))
np.max(n_t),np.mean(n_t)
ck_m=CountVectorizer()
ck_d=ck_m.fit_transform(t_d['review'])
ck_d
ck_d.toarray()#카운터 벡터기반으로 
ck_m.vocabulary_
#'with 는 특별한 단어일수도 아닐수도있음. 연관성이 높을수도 아닐수도
```

---

- EDA로 고유의 문제를 해결 해야함
- 성능 최적화 , 배치 사이즈가 클 수록 좋으나, 컴퓨터 메모리를 고려해야함
- 생성, 범위(scale) 조절, 
- 주요성분 분석, 가중치를 결정할때 그리고 시각화 할때
- 학습률= 크다고 좋은것도 아니고 작다고 좋은게 아니다. 크면 발산될 가능성 높음.신경망의 깊이
- 과소적합일어날때=>학습률 상승,에포치 상승 시켜보기
- 활성화 함수의 변경은 신중해야한다. 함수에 따라 기울기 변화가 심함
- 배치와 에포크
    - 배치가늘어나면 쪼개는 범위가 늘어남
    - 에포크: 전체 데이터를 학습 하는 수, 결론은 둘다 적절하게 조절
- 최적화,손실 함수: 일반적으로 경사하강법을 많이사용, 아담도 많이 사용
- 네트,워크 구성
    - 은닉층의 뉴런이 늘어날 수록 가중치가 늘어난다. = 수식이 늘어남
    - 뉴럴이 늘어난다는것은 네트워크가 넓어짐,층이 깊을 수록 활성화 함수를 많이 거치기 때문에 출력되는 데이터 수가 적어짐. 기억하는 특징이 늘어남!!
- 앙상블을 이용한 성능 최적화
    - 최적화를 위해 하이퍼파라미터 사용
    - 하이퍼 파라미터로 배치 정규화, 드롭아웃, 조기종료가 있다.
    - 정규화:특성 범위를 조정하는 스케일링
- 배치 정규화: 기울기 소멸과 기울기 폭발같은 문제를 해결하기 위해 사용


```py
#배치 정규화 적용해보기
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data =load_iris()
df=pd.DataFrame(data.data,columns=data.feature_names)
df = df.astype(float)#실수 데이터로 변환

df['y'] = data.target
df['y'] = df.y.replace(dict(enumerate(data.target_names)))
df
Y=pd.get_dummies(df['y'],prefix='Y')#원핫
df=pd.concat([df,Y],axis = 1)#조인 연산
df.drop(['y'],axis=1,inplace=True)
df

X_data=df[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]
X_data=np.asarray(X_data)
Y_data=df[['Y_setosa','Y_versicolor','Y_virginica']]
Y_data=np.asarray(Y_data)
X_data.shape,Y_data.shape#((150, 4), (150, 3))

#모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization

m = Sequential()
m.add(Dense(64,input_shape=(4,),activation='relu'))
m.add(Dense(128,activation='relu'))
m.add(Dense(128,activation='relu'))
m.add(Dense(64,activation='relu'))
m.add(Dense(64,activation='relu'))
m.add(Dense(3,activation='softmax'))
m.summary()
t_x,tt_x,t_y,tt_y = train_test_split(X_data,Y_data,test_size=0.2,random_state=1)#검증데이터는 보통 8:2로 나누어짐,여기서 검증데이터를 따로 안나누닌깐 걍 사이즈조절
t_x.shape#(120, 4)
#학습
m.compile(optimizer='adam', loss='categorical_crossentropy'
         ,metrics=['accuracy'])
hy = m.fit(t_x,t_y,validation_split=0.2,epochs=1000,batch_size=40,verbose=2)

import matplotlib.pyplot as plt
plt.plot(hy.history['accuracy'])
plt.plot(hy.history['val_accuracy'])
plt.plot(hy.history['loss'])
plt.plot(hy.history['val_loss'])#검증 손실값이 미쳐날뛴다.
m.evaluate(tt_x,tt_y)#[0.6256187558174133, 0.9333333373069763]

#배치 정규화
from tensorflow.keras.initializers import RandomNormal, Constant
m2 = Sequential()
m2.add(Dense(64,input_shape=(4,),activation='relu'))
m2.add(BatchNormalization())
m2.add(Dense(128,activation='relu'))
m2.add(BatchNormalization())
m2.add(Dense(128,activation='relu'))
m2.add(BatchNormalization())
m2.add(Dense(64,activation='relu'))
m2.add(BatchNormalization())
m2.add(Dense(64,activation='relu'))
m2.add(BatchNormalization(momentum=0.95,epsilon=0.005
                          ,beta_initializer=RandomNormal(mean=0.0,stddev=0.05),
                         gamma_initializer=Constant(value=0.9)))
m2.add(Dense(3,activation='softmax'))
m2.summary()
#params들이 배치정규화로 
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_6 (Dense)              (None, 64)                320       
_________________________________________________________________
batch_normalization (BatchNo (None, 64)                256       
_________________________________________________________________
dense_7 (Dense)              (None, 128)               8320      
_________________________________________________________________
batch_normalization_1 (Batch (None, 128)               512       
_________________________________________________________________
dense_8 (Dense)              (None, 128)               16512     
_________________________________________________________________
batch_normalization_2 (Batch (None, 128)               512       
_________________________________________________________________
dense_9 (Dense)              (None, 64)                8256      
_________________________________________________________________
batch_normalization_3 (Batch (None, 64)                256       
_________________________________________________________________
dense_10 (Dense)             (None, 64)                4160      
_________________________________________________________________
batch_normalization_4 (Batch (None, 64)                256       
_________________________________________________________________
dense_11 (Dense)             (None, 3)                 195       
=================================================================
Total params: 39,555
Trainable params: 38,659
Non-trainable params: 896
_________________________________________________________________
'''
m2.compile(optimizer='adam', loss='categorical_crossentropy'
         ,metrics=['accuracy'])
hy2 = m2.fit(t_x,t_y,validation_split=0.2,epochs=1000,batch_size=40,verbose=2)
import matplotlib.pyplot as plt
plt.plot(hy2.history['accuracy'])
plt.plot(hy2.history['val_accuracy'])
plt.plot(hy2.history['loss'])
plt.plot(hy2.history['val_loss'])#미쳐날뛰다가 다시 줄어든다.
m2.evaluate(tt_x,tt_y)
#[0.19030728936195374, 0.9666666388511658]


```
## 텍스트 분류
- 워드 팝콘 문제 활용
1. 캐글데이터 불러오기
2. EDA
3. 데이터 정제(문장 부호 제거, 불용어 제거, 단어 최대길이 설정, 패딩, 벡터 표상화)
4. 모델링

```py
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud,STOPWORDS
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
t_d=pd.read_csv("IMDB Dataset (1).csv")
t_d.isna().sum()#결측치확인

t_d
#불용어 설정
st=set(stopwords.words('english'))
p_ck=list(punctuation)#특수문자 리스트를 불용어 리스트에 적용
st.update(p_ck)

def 웹문서_처리(t):
    su=BeautifulSoup(t,'html.parser')
    return su.get_text()
def 정규표현_정리(t):
    return re.sub('\[[^]*\]','',t)#해당 데이터 삭제
def 불필요_정리(t):
    return re.sub(r'http\S+','',t)#해당데이터 삭제
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
t_d['review']=t_d['review'].apply(모두_정리)
#td()리뷰 문장을 매개변수로하고 함수 적용

t_d.sentiment.replace('positive',1,inplace=True)#label 값을 0,1로 변경
t_d.sentiment.replace('negative',0,inplace=True)

wc=WordCloud(max_words=2000).generate(' '.join(t_d[t_d.sentiment==1].review))
plt.imshow(wc)#긍정 워드클라우드

wc=WordCloud(max_words=2000).generate(' '.join(t_d[t_d.sentiment==0].review))
plt.imshow(wc)#부정적 워드클라우드

t_l_0=t_d[t_d.sentiment==0]['review'].str.len()
t_l_1=t_d[t_d.sentiment==1]['review'].str.len()

fig,(ax1,ax2)=plt.subplots(1,2)
ax1.hist(t_l_0)#부정
ax2.hist(t_l_1)#긍정
plt.show()

#문장별 단어의 수는 조건x 단어의 숫자가 긍정 부정을 판가름을 할수없다
t_l_0=t_d[t_d.sentiment==0]['review'].str.split().map(lambda x: len(x))
t_l_1=t_d[t_d.sentiment==1]['review'].str.split().map(lambda x: len(x))
fig,(ax1,ax2)=plt.subplots(1,2)
ax1.hist(t_l_0)#부정 
ax2.hist(t_l_1)#긍정
plt.show()#부정표가 왼쪽에 분포가많이됨그러나 긍정의 길이다 더길다.. 표만봐선 길이가 길다고 무조건 부정도 긍정도 아닌거같다.


#평균 단어 길이 조건 X
fig,(ax1,ax2)=plt.subplots(1,2)
t_l_0=t_d[t_d.sentiment==0]['review'].str.split().apply(lambda x:
                                                        [len(i) for i in x])
sns.distplot(t_l_0.map(lambda x: np.mean(x)),ax=ax1)
t_l_1=t_d[t_d.sentiment==1]['review'].str.split().apply(lambda x:
                                                        [len(i) for i in x])
sns.distplot(t_l_1.map(lambda x: np.mean(x)),ax=ax2)

def N_그램_표현(t,n,g):
    tv = CountVectorizer(ngram_range=(g,g)).fit(t)
    BoW=tv.transform(t)
    sum_Bow=BoW.sum(axis=0)
    w_f = [(w,sum_Bow[0,i])for w,i in tv.vocabulary_.items()]
    w_f=sorted(w_f,key=lambda x : x[1],reverse=True)
    return w_f[:n]
N_그램_표현(t_d.review,20,1)
'''
[('movie', 61496),
 ('film', 55088),
 ('one', 45067),
 ('like', 37303),
 ('would', 23815),
 ('even', 23720),
 ('good', 23475),
 ('really', 21806),
 ('see', 20906),
 ('get', 17692),
 ('much', 17294),
 ('story', 16812),
 ('also', 15775),
 ('time', 15660),
 ('first', 15475),
 ('great', 15475),
 ('people', 15036),
 ('make', 15030),
 ('could', 14929),
 ('made', 13562)]
'''
```