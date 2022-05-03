## 신경망 퍼셉트론 TIL 17 이지만 그냥 챕터라 이해하면됨

```py
from sklearn.linear_model import Perceptron #신경망 퍼셉트론 클래스
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,0,0,1]
p = Perceptron(tol=1e-3,random_state= 10)#종료값 매개변수 설정
p.fit(X,y)

#뉴런 만들어보기 ^^ y=wx - b
#뉴런의 계단함수 = 활성함수
def n_f(in_data):
    global w #가중치
    global b #임계값
    at_f = b# 액티베이션 함수
    for i in range(2):
        at_f +=w[i]*in_data[i]
    if at_f >=0.0:# 함수값이 0이상이면 
        return 1#참반환
    else:
        return 0#거짓반환

X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,0,0,1]
w =[0.0,0.0]#가중치
b = 0.0 #임계값, 절편이기도하고 w0이기도함
n_f(X[0]),n_f(X[1]),n_f(X[2]),n_f(X[3])#함수값입력
'''(1, 1, 1, 1)
'''

```

```py

def t_f(X,y,l_r,epch):
    global w
    global b
    for en in range(epch):
        sum_e = 0.0
        for r,t in zip(X,y):
            at = n_f(r)#예측값
            err = t - at #오차 정답값에 오차값을 빼다
            b = b+l_r*err # 임계값갱신/
            sum_e += err**2
            for i in range(2): #가중치 2개
                w[i] = w[i]+l_r*err*r[i]# 오차값은 가중치값으로 갱신
            print(w,b)
        print(f'에포크 = {en}, 학습률 = {l_r}, 에러 = {err}')
    return w
#data 준비
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,1,1,1]
w =[0.0,0.0]#가중치
b = 0.0 #임계값, 절편이기도하고 w0이기도함
l_r=0.1 #학습률
e = 5 #반복수
w = t_f(X,y,l_r,e)

n_f(X[0]),n_f(X[1]),n_f(X[2]),n_f(X[3])#함수값입력
'''(0, 1, 1, 1)
'''
```
---
### 앤드연산
```py

def AND(X):
    and_w = [0.2,0.1]
    and_b = -0.20000000000000004
    at_f=and_b
    for i in range(2):
        at_f +=and_w[i]*X[i]
    if at_f >=0.0:# 함수값이 0이상이면 
        return 1#참반환
    else:
        return 0

AND(X[0]),AND(X[1]),AND(X[2]),AND(X[3])
#(0, 0, 0, 1)#둘다 참이면 참
```
### OR연산
```py
def OR(X):
    or_w = [0.1,0.1]
    or_b = -0.1
    at_f=or_b
    for i in range(2):
        at_f +=or_w[i]*X[i]
    if at_f >=0.0:# 함수값이 0이상이면 
        return 1#참반환
    else:
        return 0

OR(X[0]),OR(X[1]),OR(X[2]),OR(X[3])
#(0, 1, 1, 1) 둘중 하나만 참이면 참
```
### XOR 연산
```py
def XOR(X):
    o_1 =  not AND(X)
    o_2 = OR(X)
    return AND([o_1,o_2])

XOR(X[0]),XOR(X[1]),XOR(X[2]),XOR(X[3])
#(0, 1, 1, 0)

```
---
신경망을 만들면 입력층 은닉층 출력층이 존재한다
파이썬으로 구현한 신경망 함수(얼추)
```py
import numpy as np
def actf(x):
    return 1/(1+np.exp(-x))#시그모이드 함수
def d_actf(x):
    return x*(1-x)#시그모이드 함수를 미분 
# y = wx + b
w=np.array([[1,2,3],
            [3,4,5]])
x=np.array([[4,5],
            [6,7],
           [8,9]])
w.dot(x)

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0],[1],[1],[0]])
in_n = 3#3개입력 층
h_n = 6 #은닉층
out_n = 1# 출력 층
np.random.seed(5)
# 초기값
w0 = 2*np.random.random((in_n,h_n))-1
w1 = 2*np.random.random((h_n,out_n))-1

for i in range(10000):
    l0 = X#1층
    # 순 전 파
    #입력*W0
    net1 =np.dot(l0,w0)
    l1=actf(net1)
    l1[:,-1] = 1#은닉층
    net2 = np.dot(l1,w1)
    l2 = actf(net2)#결과
    #역전파 알고리즘적용
    l2_e = l2-y#오차
    l2_d=l2_e*d_actf(l2)#미분(출력단의 델타값)
    
    l1_e = np.dot(l2_d,w1.T)#은닉 오차
    l1_d=l1_e*d_actf(l1)#미분 (은닉단의 델타값)
    #가중치 변화를 적용
    w1 += -0.2*np.dot(l1.T,l2_d)
    w0 += -0.2*np.dot(l0.T,l1_d)
    if i ==10:
        print(l2)
        print()
    if i ==100:
        print(l2)
        print()
    if i ==1000:
        print(l2)
        print()
    if i ==10000:
        print(l2)
        print()
    
    #print(l2)
print(l2)
#y 값 = 0 1 1 0

```
- 다른방법
```py
import numpy as np
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0],[1],[1],[0]])
in_n = 3#3개입력 층
h_n = 6 #은닉층
out_n = 1# 출력 층
np.random.seed(5)
# 초기값
w0 = 2*np.random.random((in_n,h_n))-1 #0
w1 = 2*np.random.random((h_n,out_n))-1
X.shape , w0.shape, X.dot(w0).shape,w1.shape,X.dot(w0).dot(w1).shape #4 개의 값 3개의 입력정보
'''((4, 3), (3, 6), (4, 6), (6, 1), (4, 1))
'''
end=X.dot(w0).dot(w1)
(end-y).shape,w1.T.shape,(end-y).dot(w1.T).shape
'''((4, 1), (1, 6), (4, 6))
'''
```
--- 
- 이렇게 이해하면 되지만 모듈을 쓰면 간단하다
### 케라스 모듈 
- 단층(1층) 으로 쌓기
```py
from tensorflow import keras#모듈불러옴
import numpy as np
(t_x,t_y),(tt_x,tt_y) = keras.datasets.fashion_mnist.load_data()#패션데이터 불러옴

s_t_x = t_x/255.0
s_t_x = s_t_x.reshape(-1,28*28)
s_t_x.shape# (60000,784) 784개의 피처가 존재

from sklearn.linear_model import SGDClassifier #경사하강법
from sklearn.model_selection import cross_validate#교차검증법
sc = SGDClassifier(loss = 'log',max_iter=5, random_state=42)#매개변수지정
scr = cross_validate(sc, s_t_x,t_y)
np.mean(scr['test_score'])
import tensorflow as tf

from sklearn.model_selection import train_test_split
t_x, v_x, t_y, v_y = train_test_split(s_t_x,t_y ,test_size=0.2, random_state=42)
#입력 출력 갯수를 정하자=10개, 활성화함수= 다중분류는 소프트 맥스 ,단일은 시그모이드
dense = keras.layers.Dense(10,activation='softmax',input_shape=(784,))
#단층으로 쌓음
model = keras.Sequential(dense)
#컴파일
model.compile(loss='sparse_categorical_crossentropy',metrics='accuracy
#완성된 모델 학습
model.fit(t_x,t_y,epochs=10)
#검증 데이터로 검증
model.evaluate(v_x,v_y)
```
- 여러층으로 쌓아보기
```py
from tensorflow import keras
import numpy as np
(t_x,t_y),(tt_x,tt_y) = keras.datasets.fashion_mnist.load_data()
s_t_x = t_x/255.0
s_t_x = s_t_x.reshape(-1,28*28)
from sklearn.model_selection import train_test_split
t_x, v_x, t_y, v_y = train_test_split(s_t_x,t_y ,test_size=0.2, random_state=42)
t_x.shape#(48000, 784)

#레이어 쌓기 1번방법
dense1 = keras.layers.Dense(100,activation='sigmoid',input_shape=(784,))#1번레이어
dense2 = keras.layers.Dense(10,activation='softmax')#2번레이어 softmax 는 말단과 연결
model =keras.Sequential([dense1,dense2])#2개의 레이어를 합침
model.summary() #잘 연결됬는지 확인

#레이어 쌓기 2번 방법
model =keras.Sequential([
    keras.layers.Dense(100,activation='sigmoid',input_shape=(784,),name='hidden')
    , keras.layers.Dense(10,activation='softmax',name = 'output')
],name='ck')
model.summary()#잘 연결됬는지 확인

#레이어를 쌓은후(1~2번) 컴파일을하고 학습을하면 끝
model.compile(loss='sparse_categorical_crossentropy',metrics='accuracy')
model.fit(t_x,t_y,epochs=5)
```
- 간단하게 레이어 쌓고 컴파일
```py
model = keras.Sequential()#모델 생성
model.add(keras.layers.Flatten(input_shape=(28,28)))#add()로 간단하게 추가,입력층
model.add(keras.layers.Dense(100,activation='relu',input_shape=(784,),name='hidden'))#은닉층
model.add(keras.layers.Dense(10,activation='softmax',name = 'output'))#출력층    
model.summary()
```