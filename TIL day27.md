## EDA

1. 제시된 한국어 리뷰 data를 이용하여 탐색적 데이터 분석 (EDA)을 하시오
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud,STOPWORDS
from hanspell import spell_checker
from tqdm import tqdm
from matplotlib import font_manager, rc
font_path = r'C:\Users\student\AppData\Local\Microsoft\Windows\Fonts\NanumGothic.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from pykospacing import Spacing
from sklearn.feature_extraction.text import CountVectorizer

with codecs.open('ratings_test (1).txt',encoding='utf-8') as f:
    data = [i.split('\t') for i in f.read().splitlines()]
    m = data[0]
    data=data[1:]
p_data = pd.DataFrame(data,columns=m)#인덱스 명을 카테고리화
p_data

p_data.label.replace('1',1,inplace=True)#label 값을 0,1로 변경
p_data.label.replace('0',0,inplace=True)

from wordcloud import WordCloud#한글을 플롯에 나타내기위해 워드클라우드 사용
f=r'C:\Users\student\AppData\Local\Microsoft\Windows\Fonts\NanumGothic.ttf'
wc=WordCloud(max_words=2000,font_path=f
             ,width=1000,height=600).generate(' '.join(p_data[p_data.label==1].document))
plt.imshow(wc)#긍정적 워드 클라우드

wc=WordCloud(max_words=2000,font_path=f
             ,width=1000,height=600,background_color='white').generate(' '.join(p_data[p_data.label==0].document))
plt.imshow(wc)#부정적 워드 클라우드

stopwords=set(STOPWORDS)
stopwords.add('영화')
stopwords.add('너무')
stopwords.add('진짜')
stopwords.add('ㅋㅋ')
wc=WordCloud(stopwords=stopwords,max_words=2000,font_path=f
             ,width=1000,height=600,background_color='white').generate(' '.join(p_data[p_data.label==0].document))
plt.imshow(wc)# 불용 처리한 부정적 워드 클라우드

s_w=['은','는','이','가','를','들','에게','의','을','도','으로','만','라서','하다']
st=set(s_w)
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
p_data['document']=p_data['document'].apply(모두_정리)


data_0=p_data[p_data.label==0]['document'].str.split().map(lambda x: len(x))
data_1=p_data[p_data.label==1]['document'].str.split().map(lambda x: len(x))
fig,(ax1,ax2)=plt.subplots(1,2)
ax1.hist(data_0)#부정
ax2.hist(data_1)#긍정
plt.show()# 단어가 많다고 긍정 부정 나누긴 애매함

fig,(ax1,ax2)=plt.subplots(1,2)
data_0=p_data[p_data.label==0]['document'].str.split().apply(lambda x:
                                                        [len(i) for i in x])
sns.distplot(data_0.map(lambda x: np.mean(x)),ax=ax1)

data_1=p_data[p_data.label==1]['document'].str.split().apply(lambda x:
                                                        [len(i) for i in x])
sns.distplot(data_1.map(lambda x: np.mean(x)),ax=ax2)
#히스토그램, 단어의 길이가 길다고 긍정 부정 나누기엔 애매함 
#lambda 매개변수 : 표현식

def N_그램_표현(t,n,g):
    tv = CountVectorizer(ngram_range=(g,g)).fit(t)
    BoW=tv.transform(t)
    sum_Bow=BoW.sum(axis=0)
    w_f = [(w,sum_Bow[0,i])for w,i in tv.vocabulary_.items()]
    w_f=sorted(w_f,key=lambda x : x[1],reverse=True)
    return w_f[:n]
N_그램_표현(p_data.document,20,1)

ck1_data=dict(N_그램_표현(p_data.document,20,1))
v_df = pd.DataFrame(columns=['c_w','n'])#임의의 컬럼 2개 생성
v_df['c_w']=ck1_data.keys()
v_df['n'] = ck1_data.values()
v_df
fig = px.bar(v_df, x='n' ,y='c_w', 
             orientation='h',title='c_w_number',color='c_w')
fig.show()

ck2_data=dict(N_그램_표현(p_data.document,20,2))
v_df1 = pd.DataFrame(columns=['c_w','n'])#임의의 컬럼 2개 생성
v_df1['c_w']=list(ck2_data.keys())
v_df1['n'] = list(ck2_data.values())
v_df1

```
## 그램별 그래프
```py
t_l_0_s = p_data[p_data.label==0]['document']
t_l_1_s = p_data[p_data.label==1]['document']

t_l_0_data=dict(N_그램_표현(t_l_0_s,20,1))
v_df = pd.DataFrame(columns=['c_w','n'])
v_df['c_w']=list(t_l_0_data.keys())
v_df['n'] = list(t_l_0_data.values())
fig = px.bar(v_df, x='n' ,y='c_w', orientation='h',
             title='부정적엔그램',color='c_w')
fig.show()#엔그램

t_l_1_data=dict(N_그램_표현(t_l_1_s,20,1))
v_df = pd.DataFrame(columns=['c_w','n'])#임의의 컬럼 2개 생성
v_df['c_w']=list(t_l_1_data.keys())
v_df['n'] = list(t_l_1_data.values())
fig = px.bar(v_df, x='n' ,y='c_w', orientation='h',
             title='긍정적엔그램',color='c_w')
fig.show()#엔그램

t_l_0_data=dict(N_그램_표현(t_l_0_s,20,2))
v_df = pd.DataFrame(columns=['c_w','n'])#임의의 컬럼 2개 생성
v_df['c_w']=list(t_l_0_data.keys())
v_df['n'] = list(t_l_0_data.values())
fig = px.bar(v_df, x='n' ,y='c_w', orientation='h'
             ,title='부정적바이그램',color='c_w')
fig.show()#바이그램

t_l_1_data=dict(N_그램_표현(t_l_1_s,20,2))
v_df = pd.DataFrame(columns=['c_w','n'])#임의의 컬럼 2개 생성
v_df['c_w']=list(t_l_1_data.keys())
v_df['n'] = list(t_l_1_data.values())
fig = px.bar(v_df, x='n' ,y='c_w', orientation='h'
             ,title='긍정적바이그램',color='c_w')
fig.show()#바이그램


t_l_0_data=dict(N_그램_표현(t_l_0_s,20,3))
v_df = pd.DataFrame(columns=['c_w','n'])#임의의 컬럼 2개 생성
v_df['c_w']=list(t_l_0_data.keys())
v_df['n'] = list(t_l_0_data.values())
fig = px.bar(v_df, x='n' ,y='c_w', orientation='h'
             ,title='부정적 트리그램',color='c_w')
fig.show()#트리그램

t_l_1_data=dict(N_그램_표현(t_l_1_s,20,3))
v_df = pd.DataFrame(columns=['c_w','n'])#임의의 컬럼 2개 생성
v_df['c_w']=list(t_l_1_data.keys())
v_df['n'] = list(t_l_1_data.values())
fig = px.bar(v_df, x='n' ,y='c_w', orientation='h'
             ,title='긍정적트리그램',color='c_w')
fig.show()#트리그램


```
단어간 연관 성을 보기위해 엔그램을 씀
rnn과는 다름
각단어의 연관성을 고려하며 +-@를해야함
연관성있는 단어를 차례차례 넣는게 좋을것

```py
#모델생성
X_data = p_data['document']
Y_data = p_data['label']

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

Y_data=to_categorical(Y_data)
tk = Tokenizer(num_words=5000)
tk.fit_on_texts(X_data)
X_data= tk.texts_to_matrix(X_data,mode='tfidf')


X_data.shape,Y_data.shape

from sklearn.model_selection import train_test_split
t_x,tt_x,t_y,tt_y = train_test_split(X_data,Y_data,test_size=0.2,random_state=1)

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

m=Sequential()
m.add(Dense(2,input_shape=(5000,),activation='sigmoid'))
m.compile(optimizer='adam',loss='binary_crossentropy'
          ,metrics=['accuracy'])
hy=m.fit(t_x,t_y,epochs=10,validation_data=(tt_x,tt_y))
ec=range(1,len(hy.history['accuracy'])+1)
plt.plot(ec,hy.history['loss'])
plt.plot(ec,hy.history['val_loss'])



from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import load_model
es=EarlyStopping(monitor='val_loss',mode='min',verbose=2,patience=4)
m=Sequential()
m.add(Dense(2,input_shape=(5000,),activation='sigmoid'))


m.compile(optimizer='adam',loss='binary_crossentropy'
          ,metrics=['accuracy'])
hy=m.fit(t_x,t_y,epochs=100,batch_size=64,
         validation_data=(tt_x,tt_y),callbacks=[es])

ec=range(1,len(hy.history['accuracy'])+1)
plt.plot(ec,hy.history['loss'])
plt.plot(ec,hy.history['val_loss'])

#배치정규화 사용
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.layers import BatchNormalization
es=EarlyStopping(monitor='val_loss',mode='min',verbose=2,patience=4)
m2 = Sequential()
m2.add(Dense(2,input_shape=(5000,),activation='relu'))
m2.add(BatchNormalization())
m2.add(Dense(64,activation='relu'))
m2.add(BatchNormalization())
m2.add(Dense(128,activation='relu'))
m2.add(BatchNormalization(momentum=0.95,epsilon=0.005
                          ,beta_initializer=RandomNormal(mean=0.0,stddev=0.05),
                         gamma_initializer=Constant(value=0.9)))
m2.add(Dense(2,activation='sigmoid'))
m2.summary()

m2.compile(optimizer='adam', loss='categorical_crossentropy'
         ,metrics=['accuracy'])
hy2 = m2.fit(t_x,t_y,validation_data=(tt_x,tt_y),
             epochs=100,batch_size=50,verbose=2,callbacks=[es])

             #좀더 성능이 좋아짐
```