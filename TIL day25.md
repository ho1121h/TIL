# 텍스트 데이터 MLP 학습
```py
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer#토큰
from tensorflow.keras.utils import to_categorical#원핫인코딩(정수형 범주)에만
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
import numpy as np
#데이터 로드
X_data= pd.read_csv('X_data.csv')
Y_data=pd.read_csv('Y_data.csv')#보캐뷸러데이터임

X_data.shape,Y_data.shape
#((10000, 10000), (10000, 103))
#학습데이터 분할
t_x,tt_x,t_y,tt_y=train_test_split(X_data,Y_data)

# MLP 다중뉴럴네트워크
#피드 포워드 신경망, 데이터가 영향을안줌
m=Sequential()
m.add(Dense(256,input_shape=(10000,),activation='relu'))
m.add(Dropout(0.5))
m.add(Dense(128,activation='relu'))
m.add(Dropout(0.5))
m.add(Dense(103,activation='softmax'))#다중 데이터를 처리하지만, 확률을 도출
m.compile(optimizer='adam',loss='categorical_crossentropy',
          metrics=['accuracy'])
hy=m.fit(t_x,t_y,epochs=10,validation_data=(tt_x,tt_y))

ec=range(1,len(hy.history['accuracy'])+1)#그래프 길이를 반복횟수만큼..
plt.plot(ec,hy.history['loss'])
plt.plot(ec,hy.history['val_loss'])#엥 발산하네... 
```
---
## 다시 학습
```py

d = open('k_stop_w.txt', 'r',encoding='utf-8').read()#연결 후 리드
stop=d.split('\n')
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer#토큰
from tensorflow.keras.utils import to_categorical#원핫인코딩(정수형 범주)에만
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
import numpy as np
import codecs
with codecs.open('ratings_test.txt',encoding='utf-8') as f:
    data = [i.split('\t') for i in f.read().splitlines()]
    m = data[0]
    data=data[1:]# 쓸모없는 인덱스 걸름
m#걸럿던 인덱스

p_data = pd.DataFrame(data,columns=m)#인덱스 명을 카테고리화
p_data

#토큰화 처리를 위해 불용어,설정

X=p_data['document'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣]','')# 불용어 처리
Y=p_data['label']


#데이터 전처리(패깅)
tk = Tokenizer(num_words=5000)
tk.fit_on_texts(X)#토큰화 학습
tk.word_index

X_data=tk.texts_to_matrix(X,mode='tfidf')#정규화
X_data

Y_data = to_categorical(Y)

t_x,tt_x,t_y,tt_y=train_test_split(X_data,Y_data,random_state=1)
t_x.shape,t_y.shape,tt_x.shape,tt_y.shape
#((37500, 5000), (37500, 2), (12500, 5000), (12500, 2))

#모델생성 학습
from tensorflow import keras
sgd = keras.optimizers.SGD(learning_rate=0.1)
m = Sequential()
m.add(Dense(256, input_shape=(5000,),activation='sigmoid'))#y값이 2개라 시그모이드분류
m.add(Dropout(0.5))
m.add(Dense(128, activation='relu'))
m.add(Dropout(0.5))
m.add(Dense(2, activation='softmax'))
m.compile(optimizer='adam',loss='categorical_crossentropy'
          ,metrics=['accuracy'])
hy=m.fit(t_x,t_y,epochs=10,validation_data=(tt_x,tt_y))
#시각화 및 검증..
ec=range(1,len(hy.history['accuracy'])+1)
plt.plot(ec,hy.history['loss'])
plt.plot(ec,hy.history['val_loss'])#loss가 줄어들다가 말아버림..
```
