# Prods 관련
- 결측치 수 확인
```py
data.isnull().sum().sum()
```
- 결측치가 포함된 행 수 
```py
data.isna().any(axis=1).sum()
```
- 결측치에 0치환
```py
data.fillna(0)
```
- 상관 계수
```py
q2 = data.corr()<조건문>


```
- 회귀계수
```py
 #매출액을 종속변수, TV, Radio, Social Media의 예산을 독립변수로 하여 회귀분석을
# 수행하였을 때, 세 개의 독립변수의 회귀계수를 큰 것에서부터 작은 것 순으로

from sklearn.linear_model import LinearRegression
vl=['TV', 'Radio', 'Social_Media']
lm1 = LinearRegression().fit(q3[vl],q3['Sales'])
dir(lm1)
q3_out = pd.Series(lm1.coef_,index=vl ).sort_values(ascending=False)
(np.trunc(q3_out*1000)/1000)
```
- 주어진 조건을 전체에 대한 비율 구하기
```py
q1 = data2[['컬럼1', '컬럼2', '컬럼3']].value_counts(normalize=True)

q1.index

q1[('컬럼1 타겟', '컬럼2 타겟', '컬럼3 타겟')] # 이조건으로 전체에대한 비율이 나옴

```
- np.where : replace 와 비슷하다
    - np.where(A == B, 10,11)  가주어지면  참이면 10, 거짓이면 11 로 치환함
```
Age, Sex, BP, Cholesterol 및 Na_to_k 값이 Drug 타입에 영향을 미치는지 확인하기
# 위하여 아래와 같이 데이터를 변환하고 분석을 수행하시오. 
# - Age_gr 컬럼을 만들고, Age가 20 미만은 ‘10’, 20부터 30 미만은 ‘20’, 30부터 40 미만은
# ‘30’, 40부터 50 미만은 ‘40’, 50부터 60 미만은 ‘50’, 60이상은 ‘60’으로 변환하시오. 
# - Na_K_gr 컬럼을 만들고 Na_to_k 값이 10이하는 ‘Lv1’, 20이하는 ‘Lv2’, 30이하는 ‘Lv3’, 30 
# 초과는 ‘Lv4’로 변환하시오.
```
```py
q2['Age_gr'] = np.where(q2.Age < 20, 10 , # 20 미만은 10으로
                      np.where(q2.Age < 30, 20,# 30 미만은 20
                         np.where(q2.Age < 40, 30,#40 미만은 30 
                            np.where(q2.Age < 50, 40,#50 미만은 40
                               np.where(q2.Age < 60, 50, 60 )))))#60미만은 50, 그이상 60

# - Na_K_gr 컬럼을 만들고 Na_to_k 값이 10이하는 ‘Lv1’, 20이하는 ‘Lv2’, 30이하는 ‘Lv3’, 30 
# 초과는 ‘Lv4’로 변환하시오.
q2['Na_K_gr'] = np.where(q2.Na_to_K <=10, 'lv1' ,
                         np.where(q2.Na_to_K <=20, 'lv2' ,
                                  np.where(q2.Na_to_K <=30, 'lv3' ,'lv4')))# 30초과는

```
- 카이스퀘어 검정

```py
#2. 카이스퀘어 검정
#- 1.변수별로 작업(X, Drug)
#- 2.교차표 작성
#- 3.카이스퀘어 검정 수행
var_list =['Sex', 'BP', 'Cholesterol', 'Age_gr', 'Na_K_gr']

from scipy.stats import chi2_contingency # 카이스퀘어 검정 모듈
chi2_out=[]
for i in var_list:
    temp= pd.crosstab(index = q2[i], columns=q2['Drug']) # 변수간의 교차표
    q2_out = chi2_contingency(temp)
    pvalue=q2_out[1]
    chi2_out.append([i,pvalue])


# - 검정 수행 결과, 'Drug 타입과 연관성이 있는 변수는 몇 개인가?'

chi2_out = pd.DataFrame(chi2_out,columns=['var','pvalue'])

len(chi2_out[chi2_out.pvalue <0.05]) #대립가설을 만족하는 = 연관성있는 변수조건

# 귀무가설 = H0 로 표현 : 두변수가 독립이다.[서로 상관이 없다]
# 대립가설 = H1 로 표현 : 두변수가 독립이 아니다.[상관이있다]

chi2_out[chi2_out.pvalue <0.05]['pvalue'].max() # 가장 큰 p-value

```

- A와 B 컬럼 사이의 비율에 대해 표준편차 밖의 경우를 이상치로 판단 하고  N 표준편차에 벗어난 갯수는?

```py
df # 가상의 데이터프레임이 주어짐
list = ['A', 'B'] 

#둘을 비교한 컬럼 새로 생성 서로의 비율

df['ratio'] = df[list]['A'] / df[list]['B']

평균 = df['ratio'].mean()
편차 = df['ratio'].std()

LB = 평균 - (3 * 편차)
RB = 평균 + (3 * 편차)

((df['ratio'] < LB) | (df['ratio'] >RB)).sum()

```

- 분석을 위 해 n 연속으로 기록된 컬럼명의 데이터를 사용할려하는데, n연속으로 데이터가 기록되지않은 갯수는?
- 연속으로 기록되지않은 국가를 제거한 데이터 재생성

```py
(data.groupby('컬럼명').apply(len) < n).sum()

변수 = data.groupby('컬럼명').apply(len)

list = 변수.index[변수 >= n]

ck_df = data[data.컬럼명.isin(list)]

```