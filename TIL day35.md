# 추천시스템
- 추천 시스템의 종류 
- CBF(Content-based Filtering)
- CF (Collaborative Filtering)
    - KNN
        - 아이템 기반
        - 사용자 기반
    - MF


---
## CF 아이템 기반 인접 이웃 협업 필터링
- 예를들어 영화를 이걸 볼까 말까 할때 취향 비슷한 이들에게 물어봐서 보통 추천받지 않은가? 

### - cf - knn 
```py
import pandas as pd
import numpy as np

movies = pd.read_csv('./data_movie_lens/movies.csv')
ratings = pd.read_csv('./data_movie_lens/ratings.csv')

print(movies.shape)
print(ratings.shape)

# 9천여개 영화에 대해 사용자들(600여명)이 평가한 10만여개 평점 데이터
# 영화 정보 데이터  타이틀 /장르
print(movies.shape)
movies.head()

# 유저들의 영화 별 평점 데이터 , 영화번호/ 평점
print(ratings.shape)
ratings
```
### - 사용자 - 아이템 평점 행렬로 변환

```py
# 필요한 컬럼만 추출
ratings = ratings[['userId', 'movieId', 'rating']]
ratings

# pivot_table 메소드를 사용해서 행렬 변환 - 그냥 시험적으로
ratings_matrix = ratings.pivot_table('rating', index='userId', columns='movieId')

print(ratings_matrix.shape)
ratings_matrix

# title 컬럼을 얻기 이해 movies 와 조인 수행
# 기존 movies df 와  ratings df 와 조인
rating_movies = pd.merge(ratings, movies, on='movieId')
rating_movies

# columns='title' 로 title 컬럼으로 pivot 수행. - 실습에 주된목적
ratings_matrix = rating_movies.pivot_table('rating', index='userId', columns='title')
ratings_matrix #빈값이 많다!


# NaN 값을 모두 0 으로 변환
ratings_matrix = ratings_matrix.fillna(0)
ratings_matrix
```
**-> 사용자-아이템 행렬이 만들어졌다.**

### 영화와 영화들 간 유사도 산출

```py
# 아이템-사용자 행렬로 transpose 한다.
ratings_matrix_T = ratings_matrix.transpose()    # 전치 행렬

print(ratings_matrix_T.shape) # 9719 , 610
ratings_matrix_T.head(5)

# 영화와 영화들 간 코사인 유사도 산출
from sklearn.metrics.pairwise import cosine_similarity

item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)

# cosine_similarity() 로 반환된 넘파이 행렬을 영화명을 매핑하여 DataFrame으로 변환
item_sim_df = pd.DataFrame(data=item_sim, index=ratings_matrix.columns,
                          columns=ratings_matrix.columns)

print(item_sim_df.shape) # 9719, 9719
item_sim_df.head(3)

# Godfather와 유사한 영화 6개 확인해보기
item_sim_df["Godfather, The (1972)"].sort_values(ascending=False)[:6]

'''
title
Godfather, The (1972)                        1.000000
Godfather: Part II, The (1974)               0.821773
Goodfellas (1990)                            0.664841
One Flew Over the Cuckoo's Nest (1975)       0.620536
Star Wars: Episode IV - A New Hope (1977)    0.595317
Fargo (1996)                                 0.588614
Name: Godfather, The (1972), dtype: float64
'''
# 자기 것 빼고 인셉션과 유사한 영화 5개 확인해보기
item_sim_df["Inception (2010)"].sort_values(ascending=False)[1:6]
'''
title
Dark Knight, The (2008)          0.727263
Inglourious Basterds (2009)      0.646103
Shutter Island (2010)            0.617736
Dark Knight Rises, The (2012)    0.617504
Fight Club (1999)                0.615417
Name: Inception (2010), dtype: float64
'''
```

### 여기까지가 아이템 기반 이웃필터링이고
### 다음으로 아이템 기반 인접 이웃 협업 필터링으로 개인화된 영화 추천

```py
# 평점 벡터(행 벡터)와 유사도 벡터(열 벡터)를 내적(dot)해서 예측 평점을 계산하는 함수 정의
def predict_rating(ratings_arr, item_sim_arr):
    ratings_pred = ratings_arr.dot(item_sim_arr)/ np.array([np.abs(item_sim_arr).sum(axis=1)])
    return ratings_pred
item_sim_df.head(3)
#predict_rating 함수 사용해서 내적
ratings_pred = predict_rating(ratings_matrix.values , item_sim_df.values)
ratings_pred

#그결과로 # 데이터프레임으로 변환
ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index= ratings_matrix.index,
                                   columns = ratings_matrix.columns)
print(ratings_pred_matrix.shape)
ratings_pred_matrix.head(10)

```
각 유저별로
**-> 영화 별 예측평점이 나오게 된다.**
- 예측 평점 정확도를 판단하기 위해 오차 함수인 MSE를 이용
```py
from sklearn.metrics import mean_squared_error

# 사용자가 평점을 부여한 영화에 대해서만 예측 성능 평가 MSE 를 구함. 
def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

print('아이템 기반 모든 인접 이웃 MSE: ', get_mse(ratings_pred, ratings_matrix.values ))

```
- top-n 유사도를 가진 데이터들에 대해서만 예측 평점 계산
```py
def predict_rating_topsim(ratings_arr, item_sim_arr, n=20):
    # 사용자-아이템 평점 행렬 크기만큼 0으로 채운 예측 행렬 초기화
    pred = np.zeros(ratings_arr.shape)

    # 사용자-아이템 평점 행렬의 열 크기만큼 Loop 수행. 
    for col in range(ratings_arr.shape[1]):
        # 유사도 행렬에서 유사도가 큰 순으로 n개 데이터 행렬의 index 반환
        top_n_items = [np.argsort(item_sim_arr[:, col])[:-n-1:-1]]
        # 개인화된 예측 평점을 계산
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, :][top_n_items].dot(ratings_arr[row, :][top_n_items].T) 
            pred[row, col] /= np.sum(np.abs(item_sim_arr[col, :][top_n_items]))        
    return pred


# 실행시간 2분 정도 걸림
ratings_pred = predict_rating_topsim(ratings_matrix.values , item_sim_df.values, n=20)
print('아이템 기반 인접 TOP-20 이웃 MSE: ', get_mse(ratings_pred, ratings_matrix.values ))

# 계산된 예측 평점 데이터는 DataFrame으로 재생성
ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index= ratings_matrix.index,
                                   columns = ratings_matrix.columns)
```
- 아이템 기반 인접 TOP-20 이웃 MSE:  3.6949827608772314
-> 최종적인 영화 별 예측 평점 데이터가 만들어졌다.

## 사용자에게 영화 추천을 해보자

```py

# 사용자 9번에게 영화를 추천해보자
# 추천에 앞서 9번 사용자가 높은 평점을 준 영화를 확인해보면
user_rating_id = ratings_matrix.loc[9, :]
user_rating_id[ user_rating_id > 0].sort_values(ascending=False)[:10]
'''
title
Adaptation (2002)                                                                 5.0
Citizen Kane (1941)                                                               5.0
Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)    5.0
Producers, The (1968)                                                             5.0
Lord of the Rings: The Two Towers, The (2002)                                     5.0
Lord of the Rings: The Fellowship of the Ring, The (2001)                         5.0
Back to the Future (1985)                                                         5.0
Austin Powers in Goldmember (2002)                                                5.0
Minority Report (2002)                                                            4.0
Witness (1985)                                                                    4.0
Name: 9, dtype: float64
'''
```
사용자가 관람하지 않은 영화 중에서 영화를 추천해보자, user_rating이 0보다 크면 기존에 관람한 영화라는 점을 이용해서 계산
```py
def get_unseen_movies(ratings_matrix, userId):
    # userId로 입력받은 사용자의 모든 영화정보 추출하여 Series로 반환함. 
    # 반환된 user_rating 은 영화명(title)을 index로 가지는 Series 객체임. 
    user_rating = ratings_matrix.loc[userId,:]
    
    # user_rating이 0보다 크면 기존에 관람한 영화임. 대상 index를 추출하여 list 객체로 만듬
    already_seen = user_rating[ user_rating > 0].index.tolist()
    
    # 모든 영화명을 list 객체로 만듬. 
    movies_list = ratings_matrix.columns.tolist()
    
    # list comprehension으로 already_seen에 해당하는 movie는 movies_list에서 제외함. 
    unseen_list = [ movie for movie in movies_list if movie not in already_seen]
    
    return unseen_list


# pred_df : 앞서 계산된 영화 별 예측 평점  = ratings_pred_matrix
# unseen_list : 사용자가 보지 않은 영화들
# top_n : 상위 n개를 가져온다.

def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    # 예측 평점 DataFrame에서 사용자id index와 unseen_list로 들어온 영화명 컬럼을 추출하여
    # 가장 예측 평점이 높은 순으로 정렬함. 
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_movies


# 사용자가 관람하지 않는 영화명 추출   
unseen_list = get_unseen_movies(ratings_matrix, 9)

# 아이템 기반의 인접 이웃 협업 필터링으로 영화 추천 함수사용
recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)

# 평점 데이타를 DataFrame으로 생성. 
recomm_movies = pd.DataFrame(data=recomm_movies.values, index=recomm_movies.index, columns=['pred_score'])
recomm_movies
```
결론:
아이템 기반의 인접 이웃 협업 필터링으로
사용자의 영화 예측 평점을 계산해서
상위 10개의 영화를 추천해주었다.

# MF 기반 잠재요인 협업

 행렬분해 기반의 잠재요인 협업필터링의 핵심은 사용자-아이템 평점 매트릭스 속에 숨어 있는 잠재요인을 추출해서 영화 별 예측 평점을 계산하여 영화나 아이템 추천을 가능하게 해주는 것,
 실제 평점 행렬을 행렬분해 기법으로 분해해서 잠재 요인을 추출한다.
 이 행렬 분해 기법은 유명하다.

 1. 경사 하강법을 이용한 행렬 분해
 ```py 
import numpy as np

# 원본 행렬 R 생성, 
# 분해 행렬 P와 Q 초기화, 잠재요인 차원 K는 3 설정. 
R = np.array([[4, np.NaN, np.NaN, 2, np.NaN ],
              [np.NaN, 5, np.NaN, 3, 1 ],
              [np.NaN, np.NaN, 3, 4, 4 ],
              [5, 2, 1, 2, np.NaN ]])

print(R.shape)
R # R은 4X5 행렬이다.

#Matrix Factorization
#: 행렬 R을 행렬 P, Q로 분해해보자

num_users, num_items = R.shape

K=3  # 잠재 요인은 3개

print(num_users) # M  = 4
print(num_items) # N  = 5

# P, Q 찾기
# P와 Q 매트릭스의 크기를 지정하고 정규분포를 가진 random한 값으로 입력합니다.
np.random.seed(1)

P = np.random.normal(scale=1./K, size=(num_users, K))  # 4X3 P행렬
Q = np.random.normal(scale=1./K, size=(num_items, K))  # 5X3 Q행렬
# (4, 3) * (5, 3)T -> (4, 5)

# 행렬 P, Q 초기화 상태 
print(P,'\n')# 4X3 P행렬
print(Q)# 5X3 Q행렬
 ```
 비용계산 함수를 생성하고,
분해된 행렬 P와 Q.T를 내적하여 예측 행렬을 생성한다.

실제 행렬에서 null이 아닌 위치 값만 에측 행렬의 값과 비교하여 RMSE(오차) 값을 계산하고 반환한다.

```py
from sklearn.metrics import mean_squared_error

# 실제 행렬 R과 예측 행렬 간 오차(RMSE)를 구하는 함수
# R 행렬에서 비어있지 않은 값 : non_zeros
def get_rmse(R, P, Q, non_zeros):
    error = 0
    # 두개의 분해된 행렬 P와 Q.T의 내적으로 예측 R 행렬 생성
    full_pred_matrix = np.dot(P, Q.T)
    
    # 실제 R 행렬에서 널이 아닌 값의 위치 인덱스 추출하여 실제 R 행렬과 예측 행렬의 RMSE 추출
    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]
      
    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)
    
    return 

#경사하강법에 기반하여 P와 Q 원소들을 업데이트 수행
# R > 0 인 행 위치, 열 위치, 값을 non_zeros 리스트에 저장. 
non_zeros = [ (i, j, R[i,j]) for i in range(num_users) for j in range(num_items) if R[i,j] > 0 ]
non_zeros
```
``` py
# 경사하강법
steps=10000
learning_rate=0.01
r_lambda=0.01

# P와 Q 매트릭스를 계속 업데이트(확률적 경사하강법)
for step in range(steps):  # 10000회 업데이트
    for i, j, r in non_zeros:
        
        # 실제 값과 예측 값의 차이인 오류 값 구함
        eij = r - np.dot(P[i, :], Q[j, :].T)
        
        # Regularization을 반영한 SGD(확률적 경사하강법) 업데이트 공식 적용
        P[i,:] = P[i,:] + learning_rate * ( eij * Q[j,:] - r_lambda*P[i,:] )
        Q[j,:] = Q[j,:] + learning_rate * ( eij * P[i,:] - r_lambda*Q[j,:] )

    rmse = get_rmse(R, P, Q, non_zeros)
    if (step % 50) == 0 :
        print("### iteration step : ", step," rmse : ", rmse)
```
실제 행렬과 예측 행렬 간 오차를 최소화하는 방향(rmse 감소)으로 경사하강법 진행
-> P와 Q 행렬이 업데이트 된다.
```py
pred_matrix = np.dot(P, Q.T)
print('예측 행렬:\n', np.round(pred_matrix, 3))

'''
예측 행렬:
 [[3.991 1.951 1.108 1.998 1.569]
 [4.23  4.978 1.074 2.987 1.005]
 [5.028 2.487 2.988 3.98  3.985]
 [4.974 2.002 1.003 2.002 1.555]]
'''
R #원래 값은 예측 행렬과 실제 행렬 값이 최대한 비슷하게 만들어진 것을 확인할 수 있다.
'''
array([[ 4., nan, nan,  2., nan],
       [nan,  5., nan,  3.,  1.],
       [nan, nan,  3.,  4.,  4.],
       [ 5.,  2.,  1.,  2., nan]])
'''
```