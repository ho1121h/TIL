# 비정형 텍스트 분석 정형화->분류문제 해결

- datasets 으로 데이터를 가져올때, tfdif로 정규화 할 필욘없다 왜냐하면, 이미 되있기때문... 

```py
#학습의 목적으로 구성된 데이터 셋 로드
from sklearn.datasets import *
import seaborn as sns
import pandas as pd
iris =load_iris()
data=pd.DataFrame(iris.data,columns=iris.feature_names)
sy = pd.Series(iris.target,dtype='category')
sy=sy.cat.rename_categories(iris.target_names)
data['Y'] = sy 
sns.pairplot(data)
#플롯형성
```
- 데이터셋에 등록된 와인 데이터 로드 (정형화 안된 데이터)
```py
w_data = load_wine()
w_data.feature_names,w_data.target_names
'''
피처 이름
(['alcohol',
  'malic_acid',
  'ash',
  'alcalinity_of_ash',
  'magnesium',
  'total_phenols',
  'flavanoids',
  'nonflavanoid_phenols',
  'proanthocyanins',
  'color_intensity',
  'hue',
  'od280/od315_of_diluted_wines',
  'proline'],
  y데이터 이름
 array(['class_0', 'class_1', 'class_2'], dtype='<U7'))
'''
#데이터 프레임 형성하되, w_data의 피처로 구성+컬럼명 부여
#데이터프레임에 열로 y값 추가 
w_df=pd.DataFrame(w_data.data,columns=w_data.feature_names)
sy = pd.Series(w_data.target,dtype='category')
sy=sy.cat.rename_categories(w_data.target_names)
w_df['Y'] = sy 
w_df
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(vars=['alcohol','flavanoids','total_phenols','proanthocyanins'],data=w_df,hue = 'Y')#hue는 기준을 설정
#sns.pairplot(w_df,hue='Y') =>육안으로 분류 잘된 데이터구분
Q1 분류기 생성 피쳐 4개만 사용 모델 동작 학습되는 로스를 시각화 하십시오.

X_data=w_df[['alcohol','flavanoids','total_phenols','proanthocyanins']].values#정수화

Y_data=w_df['Y'].replace(['class_0', 'class_1', 'class_2']
                             ,[0,1,2])
Y_data=Y_data.values # 정수화
Y_data#정수 

from tensorflow.keras.utils import to_categorical
Y_data=to_categorical(Y_data)#원핫인코딩화

from sklearn.model_selection import train_test_split#학습데이터 분류
t_x,tt_x,t_y,tt_y=train_test_split(X_data,Y_data,random_state=1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#모델생성
m=Sequential()
m.add(Dense(3, input_dim=4,activation='softmax'))
m.compile(optimizer='adam',loss='categorical_crossentropy'
          ,metrics=['accuracy'])
hy=m.fit(t_x,t_y,epochs=200,batch_size=1,validation_data=(tt_x,tt_y))#학습
ec=range(1,len(hy.history['accuracy'])+1)
plt.plot(ec,hy.history['loss'])
plt.plot(ec,hy.history['val_loss'])#검증

```
- Q2. 데이터를 가져오고 이진 분류를하시오
```py

from sklearn.datasets import *
import pandas as pd
Q1_data=load_breast_cancer()
Q1_data.target_names,Q1_data.feature_names#Y값이 2개라서 이진 분류를 하면된다.
pd.DataFrame(Q1_data.data)# 카테고리 이름도 없고,... y값도 없다

#데이터 프레임 재구축
df=pd.DataFrame(Q1_data.data,columns=Q1_data.feature_names)
sy = pd.Series(Q1_data.target,dtype='category')#데이터 프레임에 Y값 추가
sy=sy.cat.rename_categories(Q1_data.target_names)#Y 값의 이름을 정의
df['Y'] = sy #열의 이름을 재정의
df#데이터 프레임 

import seaborn as sns
import matplotlib.pyplot as plt
'''
sns.pairplot(vars=['mean radius','mean perimeter','mean concave points','worst concave points'],data=df ,hue = 'Y')
'''
for i in range(6):
    k=Q1_data.feature_names[5*i:5+(5*i)]
    sns.pairplot(vars=k,
                data=df,hue='Y')
    plt.show()

X_data=df[['mean radius','mean perimeter','mean concave points','worst concave points']].values

Y_data=df['Y'].replace(['malignant', 'benign']
                             ,[0,1])
Y_data=Y_data.values

from tensorflow.keras.utils import to_categorical
Y_data=to_categorical(Y_data)
Y_data#원핫 인코딩(정규화)

#학습데이터생성
from sklearn.model_selection import train_test_split
t_x,tt_x,t_y,tt_y=train_test_split(X_data,Y_data,random_state=1)
#모델생성
m1=Sequential()
m1.add(Dense(2, input_dim=4,activation='sigmoid'))
m1.compile(optimizer='adam',loss='categorical_crossentropy'
          ,metrics=['accuracy'])
hy1=m1.fit(t_x,t_y,epochs=200,batch_size=1,validation_data=(tt_x,tt_y))
#검증
ec=range(1,len(hy1.history['accuracy'])+1)
plt.plot(ec,hy1.history['loss'])
plt.plot(ec,hy1.history['val_loss'])

```
- Q3.
1. data load
2. 로드한 데이터를 분석하여 적용할 모델 결정
- 피쳐의 갯수는 자유
```py
df1=pd.read_csv('data1 (1).csv',encoding='latin1', index_col=0)
df1

df1.columns
#Index(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Y'], dtype='object') 피쳐,y 이름

sns.pairplot(df1,hue='Y')#유사도가 상당히 높다, 그나마 유사하지않은 피쳐 선택
#피쳐3개, y3개
X_data=df1[['1','2','7']].values#정수화 

Y_data=df1['Y'].replace(['0', '1', '2']
                             ,[0,1,2])

Y_data=Y_data.values
Y_data=to_categorical(Y_data)
Y_data#원핫인코딩

from sklearn.model_selection import train_test_split
t_x,tt_x,t_y,tt_y=train_test_split(X_data,Y_data,random_state=1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#분류 문제 풀이
#모델생성, 
m=Sequential()
m.add(Dense(3, input_dim=3,activation='softmax'))
m.add(Dense())
m.compile(optimizer='adam',loss='categorical_crossentropy'
          ,metrics=['accuracy'])
hy=m.fit(t_x,t_y,epochs=200,batch_size=1,validation_data=(tt_x,tt_y))
#결과측정

ec=range(1,len(hy.history['accuracy'])+1)
plt.plot(ec,hy.history['loss'])
plt.plot(ec,hy.history['val_loss'])
```
위 문제의 문제점
- 로스가 차이가 남(하나가 줄다가 말음)
- 과도한 학습의 문제
- 단층으로 학습해서 문제가일어남

### 텍스트를 간단히 메트릭스화
- 데이터 셋으로 가져올때 정수화가 안됬을 경우
```py
from tensorflow.keras.preprocessing.text import Tokenizer
t_data = ['신문 전자 삼성 전자 주식 신문', '삼성 신문','한화 신문','기록 신문']
tk=Tokenizer()
tk.fit_on_texts(t_data) # 토큰 학습
tk.word_index#밸리데이션화
#{'신문': 1, '전자': 2, '삼성': 3, '주식': 4, '한화': 5, '기록': 6}
tk.texts_to_matrix(t_data)#binary 단어의 유무를 정수로 표현
'''
array([[0., 1., 1., 1., 1., 0., 0.],
       [0., 1., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 1., 0.],
       [0., 1., 0., 0., 0., 0., 1.]])
'''
tk.texts_to_matrix(t_data,mode='count')#단어의 빈도수
'''array([[0., 2., 2., 1., 1., 0., 0.],
       [0., 1., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 1., 0.],
       [0., 1., 0., 0., 0., 0., 1.]])'''
tk.texts_to_matrix(t_data,mode='tfidf')#가중치 수치(문자 갯수의 대비)
'''
array([[0.        , 0.99520933, 1.8601123 , 0.84729786, 1.09861229,
        0.        , 0.        ],
       [0.        , 0.58778666, 0.        , 0.84729786, 0.        ,
        0.        , 0.        ],
       [0.        , 0.58778666, 0.        , 0.        , 0.        ,
        1.09861229, 0.        ],
       [0.        , 0.58778666, 0.        , 0.        , 0.        ,
        0.        , 1.09861229]])
'''
from sklearn.datasets import fetch_20newsgroups #데이터 셋
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

w_data = fetch_20newsgroups()
print(w_data.keys())#dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])
data = pd.DataFrame(w_data.data)
#w_data.target
len(w_data.target_names) #20종류의 11314개의 데이터
X_data=w_data.data
Y_data=w_data.target

from sklearn.model_selection import train_test_split
import numpy as np

all_data = pd.DataFrame()
all_data['X']=X_data
all_data['Y']=Y_data
all_data#내부문장의 피쳐로 다중 분류해야함
#즉 단어가 피쳐 
#최종적으로 정형화 하면끝
tk = Tokenizer(num_words=10000)#정형화를위해 토큰화
tk.fit_on_texts([X_data[0]])#대괄호로 감싸 리스트로 생성
len(tk.word_index)#94개의 단어 가 만들어진다

tk.fit_on_texts(X_data)#키 밸류 딕셔너리의 형태
t_x_data=tk.texts_to_matrix(X_data)#문장을 표현할 방법 1
t_x_data
t1_x_data=tk.texts_to_matrix(X_data,mode='count')#문장을 표현할 방법 2
t1_x_data
t2_x_data=tk.texts_to_matrix(X_data,mode='tfidf')#문장을 표현할 방법 3
t2_x_data# 단어가 얼마나 중요한지 가중치로 표현 ,
#특정 문서에서 빈도가 높거나, 전체문서에서 특정단어가 낮을 수록 값은 높아진다
Y_data = to_categorical(Y_data)
Y_data#원핫 인코딩화
X_data1=t_x_data
X_data2=t1_x_data
X_data3=t2_x_data
Y_data

t_x,tt_x,t_y,tt_y=train_test_split(X_data1,Y_data,random_state=1)

t_x_1,tt_x_1,t_y_1,tt_y_1=train_test_split(X_data1,Y_data,random_state=1)#단어의 유무 특징을 갖는 data1
#단어의 빈도수를 특징으로 갖는 data2
t_x_2,tt_x_2,t_y_2,tt_y_2=train_test_split(X_data2,Y_data,random_state=1)#카운트

#단어의 가중치를 특징을 갖는 data3
t_x_3,tt_x_3,t_y_3,tt_y_3=train_test_split(X_data3,Y_data,random_state=1)#tfidf

Y=w_data.target
#학습할 모델 생성
w_all_data = pd.DataFrame(X_data3)
w_all_data['Y']=Y
w_all_data
#피쳐가 너어어무 많아서 dropout으로 걸러주기
#다층퍼셉트론
from tensorflow.keras.layers import Dense ,Dropout
from tensorflow.keras.models import Sequential
m = Sequential()
m.add(Dense(256, input_shape=(10000,),activation='relu'))
m.add(Dropout(0.5))
m.add(Dense(128, activation='relu'))
m.add(Dropout(0.5))
m.add(Dense(20, activation='softmax'))
m.compile(optimizer='adam',loss='categorical_crossentropy'
          ,metrics=['accuracy'])
hy=m.fit(t_x_1,t_y_1,epochs=10,validation_data=(tt_x_1,tt_y_1))
ec=range(1,len(hy.history['accuracy'])+1)
plt.plot(ec,hy.history['loss'])
plt.plot(ec,hy.history['val_loss'])
```
- 윗 코드가 더러우니 다시
### 함수화로 처리

```py
from sklearn.datasets import fetch_20newsgroups 
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#data 수집
data = fetch_20newsgroups()
X_data=data.data
Y_data=data.target
#data 전처리
def 택스트_전처리기(X_data,Y_data,mode):
    tk = Tokenizer(num_words=10000)
    tk.fit_on_texts(X_data)
    X=tk.texts_to_matrix(X_data,mode=mode)
    Y= to_categorical(Y_data)
    t_x,tt_x,t_y,tt_y = train_test_split(X,Y)
    return t_x,tt_x,t_y,tt_y 
#data 모델
def 텍스트_분류기(t_all):
    m = Sequential()
    m.add(Dense(256, input_shape=(10000,),activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(20, activation='softmax'))
    m.compile(optimizer='adam',loss='categorical_crossentropy'
              ,metrics=['accuracy'])
    hy=m.fit(t_all[0],t_all[2],epochs=10,validation_data=(t_all[1],t_all[3]))
    return hy
t_all=택스트_전처리기(X_data,Y_data,mode='tfidf')#mode만 상황에 따라 바꾸면댐
t_all
hy=텍스트_분류기(t_all)
#검증
ec=range(1,len(hy.history['accuracy'])+1)
plt.plot(ec,hy.history['loss'])
plt.plot(ec,hy.history['val_loss'])
```

# 결론
- 가져온 데이터 셋을 잘보고
- 정수화안됨->정수화
- 정규화 안됨-> 정규화 
    - tk.texts_to_matrix(X_data,mode='tfidf')
- 분류로 문제를 해결할지 
    - 이진분류면 시그모이드, 다중분류면 소프트맥스
- 회귀로 문제를 해결할지 
- 에 따라서 각 상황에 따라 문제를 해결하면된다.