## 신경망 퍼셉트론 

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