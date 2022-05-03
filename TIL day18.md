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
