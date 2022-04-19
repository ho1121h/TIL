## 딥러닝 기초
- 함수와 벡터 관련
```python
import numpy as np
from numpy.linalg import inv
a = np.array([2, 1])
print(a)#백터 선
'''
[2 1]
'''
행렬1=np.array([[1,2],[3,4]])#행렬 면
print(행렬1)
'''
[[1 2]
 [3 4]]
'''
m1=np.array([[1,2],[3,4]])
m2=np.array([[1,2],[3,4]])
print(m1+m2)
'''
[[2 4]
 [6 8]]
'''
print(m1*m2)#이러면 스칼라 곱이되서 하면안된다 내적 해야된다,.
'''
[[ 1  4]
 [ 9 16]]
'''
i_m1=inv(m1)
print(i_m1)
'''
[[-2.   1. ]
 [ 1.5 -0.5]]
'''
print(np.dot(m1,m2))
'''
[[ 7 10]
 [15 22]]
'''
```

```python

import numpy as np
from numpy.linalg import inv
A= np.arange(1,10).reshape(3,3)
print(A)
'''
[[1 2 3]
 [4 5 6]
 [7 8 9]]
'''
print(inv(A))#역행렬
'''
[[-4.50359963e+15  9.00719925e+15 -4.50359963e+15]
 [ 9.00719925e+15 -1.80143985e+16  9.00719925e+15]
 [-4.50359963e+15  9.00719925e+15 -4.50359963e+15]]

'''
'''
2x - y = 0
x + y =3
Y=WX
W=YX^-1
'''
X = np.array([[2,3],[1,-2]])#연립 방정식의 x,y 앞자리
Y = np.array([[1], [4]]) #연립 방적식의 결과
X_inv=inv(X)
#print(Y.shape)
#print(X_inv.shape)
W=X_inv.dot(Y)
print(W)
'''
[[ 2.] x값
 [-1.]] y값

'''

```
--- 

```python
#1. 데이터 수집
A_길이 = [25.4,26.5,27.5,28.4,29.4,29.0,30.0,30.0,31.0,31.2]
A_무게 = [243,290,340,363,430,450,500,394,450,500]
print(len(A_길이),len(A_무게)) #10 10

import matplotlib.pyplot as plt
plt.scatter(A_길이,A_무게)

B_길이 = [5.4,6.5,7.5,8.4,9.4,9.0,7.0,1.0,1.0,1.2]
B_무게 = [43,90,40,63,30,50,50,94,50,50]
plt.scatter(B_길이,B_무게)


plt.scatter(A_길이,A_무게)
plt.scatter(B_길이,B_무게)

#2 데이터 정리
길이 = A_길이+B_길이 
무게 = A_무게+B_무게

data = [[길이, 무게]for 길이,무게 in zip(길이,무게)]
X=data

#3.모델 생성 
from sklearn.neighbors import KNeighborsClassifier#모듈사용
kn = KNeighborsClassifier()
kn.fit(X,Y)
```
---

```python

from sklearn.datasets import *
import matplotlib.pyplot as plt
import seaborn as sns
d=load_digits()
sns.heatmap(d.images[0],cmap=mpl.cm.bone_r,annot=True, cbar=True)
plt.title("확인")

plt.show()

d.images[0]#데이터화

#데이터 전처리 인코딩
d.images[0].flatten()

import pandas as pd
t=load_boston()#시몬으로 데이터 로드
df=pd.DataFrame(t.data, columns=t.feature_names)
df

df['가격'] = t.target# 정답
df

ck=sns.pairplot(df[['가격','RM','AGE','CRIM']])#각 해당 플롯 표시
plt.show()

x=t.data

y=t.target# 정답

from sklearn.linear_model import LinearRegression#모듈사용
m=LinearRegression()#모델 생성
m.fit(x, y)#학습
#Y=WX->Y=10X 에서 최적의 W값을 찾음
out_d = m.predict(t.data)#아웃풋 얻기 학습한 결과로
plt.scatter(y,out_d)
plt.show()

t1 = load_iris()#시몬에서 아이리스 로드

from sklearn.svm import SVC
f = [2,3]
X = t1.data[:,f]
Y = t1.target
m = SVC(kernel = 'linear', random_state= 0)
m.fit(X,Y)###여기 까지 학습이 끝