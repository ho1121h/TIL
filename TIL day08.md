## Numpy 연산
 연산을 통한 결과 , 주피터 노트 이용 ,행렬연산 특화
 ```python
 import numpy as np
t_a=np.arange(1,11)
t_a.sum()
# >> 55
# reshape 배열의 차원의 크기를 변경
t_a=np.arange(1,13).reshape(3,4)#첫 매개변수 값이 고차원임-> 2차원3개원소 1차원 4개원소를 가짐 
t_a
# >>array([[ 1,  2,  3,  4],
#          [ 5,  6,  7,  8],
#          [ 9, 10, 11, 12]])
t_a.sum(axis=0)#2차원
#>>array([15, 18, 21, 24])
t_a.sum(axis=1)#1차원
#>>array([10, 26, 42])
#np.vstack 결합된 배열이 만들어짐. 결합원소의 길이가 같아야함
v_a=np.array([1,2,3])
v_a_1=np.array([4,5,6])
#>>array([[1, 2, 3],
#       [4, 5, 6]])
#넘파이는 사칙연산이 가능
## 내적 (행렬의 곱) 가로와 세로는 일치해야한다!@!!!@!!
x=np.arange(1,7).reshape(2,3)#저차원이 3
y=np.arange(1,7).reshape(3,2)#저차원이 2
print(x)
print(y)
np.dot(x,y)#내적
x.dot(y)#내적 방법2
#x=[[1 2 3]
#  [4 5 6]]
#y=[[1 2]
#  [3 4]
#  [5 6]]
#array([[22, 28],
#       [49, 64]])