## 데이터의 시각화
데이터를 보기 좋게 그래프로 그려보기

### matplotlib
맷플롯 사용
해당 모듈을 사용해야 시각화 가능
```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Gulim'#한글이 출력 되게끔
matplotlib.rcParams['axes.unicode_minus']=False #음수표현을 위한 속성값 변경
```
```python
x=[1,2,3]
y=[-2,-4,-8]
y1=[1,3,7]
plt.plot(x,y) #맷플롯에 data 전달 (1,-2),(2,-4),(3,-8)에 배치된다
plt.plot(x,y1)
plt.show() #그래프를 한번 보여주면 초기화됨 
```
```python
import numpy as np #넘파이 사용
X_1=range(100)
Y_1=[np.cos(i) for i in X_1]#1~99를 코사인으로

X_2=range(100)
Y_2=[np.sin(i) for i in X_1]#1~99를 사인으로

plt.plot(X_1,Y_1)# 두개의 원소 갯수는 같아야함 
plt.plot(X_2,Y_2)

```

```python
fig, ax =plt.subplots()
X_1=range(100)
Y_1=[np.cos(i) for i in X_1]
ax.plot(X_1,Y_1) #ax 안에 들어감
ax.set(title="cos_g", xlabel='X',ylabel='Y')# set 으로 각 각 이름 붙임
plt.show()
#결과로 제목 , X축 , Y축이 이름이 형성

```

```python
x=[1,2,3]
y=[4,8,6]

plt.plot(x,y)#선 그래프
plt.title("꺾은선 그래프",fontdict={'family':"Gulim",'size':25})#제목 / 폰트 수정 및 사이즈 크기
#가로 left center rioght
plt.xlabel("가로축",color="#00aa00",fontdict={'family':"Gulim",'size':25},loc="left")
#세로 top center bottom
plt.ylabel("세로축",fontdict={'family':"Gulim",'size':25})
plt.xticks=([1,2,3]) #주어진 데이터와 일치해야함
plt.yticks=([1,2,3,5,8]) 
plt.show()

```
---
### 서브플롯

```python
fig, ax=plt.subplots(nrows=2,ncols=2)
print(ax)
#열 2개 행 2개의 칸생성
fig, ax=plt.subplots(2,2)# 4칸의 그림 생성
x=np.linspace(-1,1,100)
y1=np.sin(x)
y2=np.cos(x)
y3=np.tan(x)
y4=np.exp(x)
ax[0,0].plot(x,y1)#첫 번째 배치
ax[0,1].plot(x,y2)# 두번째 배치
ax[1,0].plot(x,y3)#세번째 배치
ax[1,1].plot(x,y4)#네번째 배치
#다음과 같이 배치도 가능
ax1=plt.subplot(221)#서브플롯(행 열 몇번)/첫번째 배치
plt.plot(x,y1)
ax2=plt.subplot(222)#2행 2열 중 2번째에 배치
plt.plot(x,y2)
ax3=plt.subplot(223)# "  "      3번째에 배치
plt.plot(x,y3)
ax4=plt.subplot(224)# "   "     4번째에 배치
plt.plot(x,y4)
plt.show()
```
---
점선 및 컬러 속성, 라벨 부여
```python
x=np.linspace(-1,1,100)
y1=np.sin(x)
y2=np.cos(x)
y3=np.tan(x)
y4=np.exp(x)

plt.plot(x,y1,c="r",ls="dashed",label="sin")
plt.plot(x,y2,c="b",ls="dashed",label="cos")
plt.plot(x,y3,c="g",ls="solid",label="tan")
plt.plot(x,y4,c="y",ls="solid",label="exp",linewidth=5)
plt.legend(loc="best",shadow=True)#그림자추가
plt.show()
###벡터
plt.plot(X,Y,marker='v')#벡터로 나타내기
plt.plot(X1,Y1,marker='v',markersize=15)



```