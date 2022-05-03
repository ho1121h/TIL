# CNN
- 컨벌루션 신경망
생명체의 영상 처리구조에서 힌트를 얻음
- 전처리,특징추출,분류

```py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.datasets.fashion_mnist import load_data
np.random.seed(42)
tf.random.set_seed(42)
import matplotlib.pyplot as plt
m = keras.models.load_model('best2_m.h5')
conv = m.layers[0]
conv.weights[0].shape, conv.weights[1].shape
'''(TensorShape([3, 3, 1, 32]), TensorShape([32]))
'''
w = conv.weights[0].numpy()
fig, axs = plt.subplots(2,16,figsize = (15,2))
for i in range(2):
    for j in range(16):
        axs[i,j].imshow(w[:,:,0,i*16+j],vmin=-0.5,vmax=0.5)
        axs[i,j].axis('off')
n_m = keras.Sequential()
n_m.add(keras.layers.Conv2D(32, kernel_size=3,activation='relu',padding='same'
                           ,input_shape=(28,28,1)))
#피쳐 출력                        
n_conv= n_m.layers[0]
n_w=n_conv.weights[0].numpy()#32게
fig, axs = plt.subplots(2,16,figsize = (15,2))
for i in range(2):
    for j in range(16):
        axs[i,j].imshow(n_w[:,:,0,i*16+j],vmin=-0.5,vmax=0.5)
        axs[i,j].axis('off')
#피쳐 출력
    
cov_act1=keras.Model(m.input,m.layers[0].output)#첫번째의 레이어를 출력으로 함
(x_data,y_data),(t_x_data,t_y_data) = keras.datasets.fashion_mnist.load_data()
in_data = x_data.reshape(-1,28,28,1)/255.0 
plt.imshow(x_data[0],cmap='gray_r')

fig, axs = plt.subplots(4,8,figsize = (15,8))#피쳐맵에 맞게 플롯수 조정
for i in range(4):
    for j in range(8):
        axs[i,j].imshow(f_map[0,:,:,i*8+j])
        axs[i,j].axis('off')
#중간 폴링층 제외한 다음층
cov_act2=keras.Model(m.input,m.layers[2].output)
f2_map = cov_act2.predict(in_data[0:1])
f2_map.shape #64개
fig, axs = plt.subplots(8,8,figsize = (8,8))
for i in range(8):
    for j in range(8):
        axs[i,j].imshow(f2_map[0,:,:,i*8+j])
        axs[i,j].axis('off')
```
---
### 목적 CNN분류기
1. 모델을 만들어 주세요
    - 컨벌루션 2층
    - 뉴런은 최소 1층
    - 형상 출력
2. 학습
    - 최적의 모델을 저장하시오
    - 조건을 부여하여 학습의 진행을 정기 시켜라
    - 검증 data를 이용하여 점수를 도출 하시오
3. 시각화
    - 테스트 data를 이용하여 입력을 통한 결과 출력을 시각화 하시오
    - 테스트 data를 이용하여 점수를 도출하시오
4. 피쳐 검토
    - 학습된 모델을 이용하여 컨벌루션 층의 피처들을 시각화 하시오.
```py 
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
np.random.seed(42)
tf.random.set_seed(42)
from sklearn.model_selection import train_test_split
(x_data,y_data),(t_x_data,t_y_data) =load_data()
'''
x_data의 구조
(60000, 28, 28)
y_data의 갯수 10개
'''
#1, 모델을 만들어 주세요
x_data = x_data.reshape(-1,28,28,1)/255.0
tt_x = t_x_data.reshape(-1,28,28,1)/255.0
tt_y = t_y_data
t_x,v_x, t_y, v_y = train_test_split(x_data,y_data,test_size=0.2,random_state=42)

m = keras.Sequential()#모델 생성
#컨벌루션 층
m.add(keras.layers.Conv2D(32,kernel_size=3,activation='relu',padding='same',
                         input_shape= (28,28,1)))#컨벌류션 1층
m.add(keras.layers.MaxPool2D(2))
m.add(keras.layers.Conv2D(64,kernel_size=(3,3),
                          activation='relu',padding='same'))#컨벌루션2층
m.add(keras.layers.MaxPool2D(2))

#뉴럴층
m.add(keras.layers.Flatten())
m.add(keras.layers.Dense(100,activation='relu'))
m.add(keras.layers.Dropout(0.4))
m.add(keras.layers.Dense(10,activation='softmax'))
m.compile(loss='sparse_categorical_crossentropy',
          optimizer='adam',metrics='accuracy')
m.summary() #형상 출력

#2. 학습
#학습 조건 (저장, 정기)
ck_p = keras.callbacks.ModelCheckpoint('best_model.h5',save_best_only=True)
e_st = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)
#학습하기
hy = m.fit(t_x,t_y, epochs=20,validation_data=(v_x,v_y),callbacks=[ck_p,e_st])


#3. 시각화
plt.plot(hy.history['loss'])
plt.plot(hy.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()
m.evaluate(v_x, v_y)#검증

end=list(np.unique(y_data))
y_l=m.predict(tt_x)
for i in range(len(y_l[100])):
    print(end[np.argmax(y_l[i:i+1])])
    plt.imshow(tt_x[i].reshape(28,28),cmap='gray_r')
    plt.show()


#4. 학습된 모델 불러오기
m_ck=keras.models.load_model('best_model.h5')
w1=m_ck.layers[0].weights[0].numpy()
w2=m_ck.layers[2].weights[0].numpy()

fig, axs = plt.subplots(2,16,figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i,j].imshow(w1[:,:,0,i*16+j])
        axs[i,j].axis('off')

fig, axs = plt.subplots(4,16,figsize=(15,2))
for i in range(4):
    for j in range(16):
        axs[i,j].imshow(w2[:,:,0,i*16+j])
        axs[i,j].axis('off')       

plt.imshow(t_x_data[0],cmap='gray_r') 

#시각화 피쳐맵
cov_act1=keras.Model(m_ck.input,
                     m_ck.layers[0].output)
f1_map=cov_act1.predict(tt_x[0:1])
fig, axs = plt.subplots(4,8,figsize=(15,8))

for i in range(4):
    for j in range(8):
        axs[i,j].imshow(f1_map[0,:,:,i*8+j])
        axs[i,j].axis('off')      
#피쳐맵 시각화 2
cov_act2=keras.Model(m_ck.input,
                     m_ck.layers[2].output)
f2_map=cov_act2.predict(tt_x[0:1])
fig, axs = plt.subplots(8,8,figsize=(12,12))
for i in range(8):
    for j in range(8):
        axs[i,j].imshow(f2_map[0,:,:,i*8+j])
        axs[i,j].axis('off')
plt.show()
```