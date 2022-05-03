## TIL 18 챕터
- 딥러닝 2일차
- 컴파일: 모델을 어떤 방식으로 학습시키는 결정하는 과정.
    - 모델 컴파일의 주요항목은 최적화 방법인 옵티마이저(optimaizer="")와 손실함수(loss="")
    - 훈련과 평가시 계산할 지표를 추가로 지정가능(metrics=[""])

- 옵티마이저 
    - Adam: 가장 단순하게 많이 사용하는 방법, 학습률을 조절할려면 opt=keras.optimizers.adam(learning_rate=0.002) 같이 써야함
    - sgd: 일부의 기울기를 추출 momentum이라는 하이퍼매개변수 조절가능
    - 이외에 다른 종류는 많지만 배운것만 적용함
    - 옵티마이저방식에 따라 최적해에 수렴하는 속도가 다름을 유의하자
```py 
import tensorflow as tf
from tensorflow import keras
import numpy as np
(t_x,t_y),(tt_x,tt_y) = keras.datasets.fashion_mnist.load_data()
s_t_x = t_x/255.0
s_tt_x=tt_x/255.0
from sklearn.model_selection import train_test_split
t_x, v_x, t_y, v_y = train_test_split(s_t_x,t_y ,test_size=0.2, random_state=42)
t_x.shape#(48000, 28, 28)

model = keras.Sequential()# 모델 생성
model.add(keras.layers.Flatten(input_shape=(28,28)))#입력층 
model.add(keras.layers.Dense(100,activation='relu',input_shape=(784,),name='hidden'))#은닉층 28*28 =784 *100 = 78500
model.add(keras.layers.Dense(10,activation='softmax',name = 'output'))          
model.summary()
model.compile(loss='sparse_categorical_crossentropy',metrics='accuracy')
model.fit(t_x,t_y,epochs=5)#학습 5회 반복
model.evaluate(v_x,v_y)# 검증

# SGD(경사하강법)으로 컴파일
sgd = keras.optimizers.SGD(learning_rate=0.1)#학습율 설정
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics='accuracy')
sgd = keras.optimizers.SGD(momentum=0.9,nesterov=True)#관성부여설정
#옵티 마이저 다른 종류
keras.optimizers.RMSprop
keras.optimizers.Adagrad#상황에 따른 이동거리조정
###
model1 = keras.Sequential()
model1.add(keras.layers.Flatten(input_shape=(28,28)))
model1.add(keras.layers.Dense(100,activation='relu',input_shape=(784,),name='hidden'))#은닉층
model1.add(keras.layers.Dense(10,activation='softmax',name = 'output'))          
model1.summary()
model1.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')#adam 으로 컴파일 후 
model1.fit(s_tt_x,t_y,epochs=5)#학습
###모델생성 후 학습 및 시각화

m1 = keras.Sequential()#모델생성!
m1.add(keras.layers.Flatten(input_shape = (28,28)))#입력층 레이어 생성
m1.add(keras.layers.Dense(100,activation='relu'))
m1.add(keras.layers.Dense(300,activation='relu'))
m1.add(keras.layers.Dense(10,activation='softmax'))#종단 뉴런 층
m1.summary()
m1.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
history = m1.fit(t_x, t_y, epochs = 30, validation_data=(v_x,v_y))#검증 데이터를 두고 학습 실행
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)#과대적합됨, 검증 로스가 점점 상승함

m2 = keras.Sequential()#모델2 생성!
m2.add(keras.layers.Flatten(input_shape = (28,28)))#입력층 레이어 생성
m2.add(keras.layers.Dense(300,activation='relu'))
m2.add(keras.layers.Dense(100,activation='relu'))
m2.add(keras.layers.Dense(10,activation='softmax'))#종단 뉴런 층
m2.summary()
m2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
history1 = m2.fit(t_x, t_y, epochs = 20, validation_data=(v_x,v_y))
pd.DataFrame(history1.history).plot(figsize=(8,5))
plt.gca().set_ylim(0,1)# 아직 과대적합됨

m3 = keras.Sequential()#모델3 생성!
m3.add(keras.layers.Flatten(input_shape = (28,28)))#입력층 레이어 생성
m3.add(keras.layers.Dense(300,activation='relu'))
m3.add(keras.layers.Dense(100,activation='relu'))
m3.add(keras.layers.Dense(10,activation='softmax'))#종단 뉴런 층
m3.summary()
m3.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics='accuracy')
history3 = m3.fit(t_x, t_y, epochs = 20, validation_data=(v_x,v_y))
pd.DataFrame(history3.history).plot(figsize=(8,5))
plt.gca().set_ylim(0,1)#정상적으로 출력
#loss 가 안떨어지면 기울기를 못찾는중일 것,
plt.plot(history3.history['loss'])#잘학습 됬는지 확인
plt.plot(history3.history['val_loss'])

```

- 생성된 모델 시각화 및 저장

```py
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(42)#랜덤으로 가져올 데이터
tf.random.set_seed(42)#가져올 데이터가 고정

data = fetch_california_housing()
in_data = StandardScaler().fit_transform(data.data)
x_data, tt_x, y_data, tt_y =train_test_split(in_data,data.target ,random_state=42)

t_x, v_x,t_y,v_y = train_test_split(x_data,y_data, random_state= 42, test_size= 0.2)
#모델 생성, 레이어 생성
m = keras.models.Sequential()
m.add(keras.layers.Flatten(input_shape=t_x.shape[1:]))
m.add(keras.layers.Dense(30,activation='relu'))
m.add(keras.layers.Dense(1))
m.summary()

keras.utils.plot_model(m,'m.png')#이미지화
keras.utils.plot_model(m,"m.png",show_shapes=True)#인풋,아웃풋값도 이미지화
#컴파일 및 학습
m.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-3),loss='mean_squared_error',metrics='accuracy')
hy=m.fit(t_x,t_y,validation_data=(v_x,v_y),epochs = 20)
t_hy = m.evaluate(tt_x,tt_y)

```
---
### 드롭아웃
- 학습한 데이터가 과적합 됬을때 해결방법
```py
#외부 모듈 수집
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
(t_x,t_y),(tt_x,tt_y) = keras.datasets.fashion_mnist.load_data()
s_t_x = t_x/255.0
s_tt_x=tt_x/255.0
from sklearn.model_selection import train_test_split
t_x, v_x, t_y, v_y = train_test_split(s_t_x,t_y ,test_size=0.2, random_state=42)
#모델 1생성
m=keras.Sequential()
m.add(keras.layers.Flatten(input_shape=t_x.shape[1:]))#784
m.add(keras.layers.Dense(100,activation='relu'))#100
m.add(keras.layers.Dense(300,activation='relu'))#300
m.add(keras.layers.Dense(10,activation='softmax'))
m.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics='accuracy')
hy = m.fit(t_x,t_y,validation_data=(v_x,v_y),epochs = 20)
plt.plot(hy.history['loss'])
plt.plot(hy.history['val_loss'])# 로스가 정상적으로 감소

# 모델 2 생성

m1=keras.Sequential()
m1.add(keras.layers.Flatten(input_shape=t_x.shape[1:]))#784
m1.add(keras.layers.Dense(100,activation='relu'))#100
m1.add(keras.layers.Dense(300,activation='relu'))#300
m1.add(keras.layers.Dense(10,activation='softmax'))
m1.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
hy = m1.fit(t_x,t_y,validation_data=(v_x,v_y),epochs = 20)
plt.plot(hy.history['loss'])
plt.plot(hy.history['val_loss'])#검증 로스가 발산함을 볼 수 있음

#뉴런이 너~~무 많다 걸러주자
m1=keras.Sequential()
m1.add(keras.layers.Flatten(input_shape=t_x.shape[1:]))#784
m1.add(keras.layers.Dense(100,activation='relu'))#100
m1.add(keras.layers.Dropout(0.3))#30퍼의 뉴런을 삭제
m1.add(keras.layers.Dense(300,activation='relu'))#300
m1.add(keras.layers.Dense(10,activation='softmax'))
m1.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
hy = m1.fit(t_x,t_y,validation_data=(v_x,v_y),epochs = 20)
plt.plot(hy.history['loss'])
plt.plot(hy.history['val_loss'])#응 정상적으로 줄어들어

```
- 가중치 저장
``` py
m1.save_weights('m_1_w.h5')#모델의 웨이트값 저장

m1.save('m1.h5')# 모델 저장
```
- 학습된 가중치,모델 불러오기
```py
m2.load_weights("m_1_w.h5")
m2.evaluate(t_x,t_y)# m1 의 학습된 모델이 불러와짐

m3 = keras.models.load_model('m1.h5')    
m3.evaluate(t_x,t_y)
```