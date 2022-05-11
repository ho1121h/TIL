# 언어모델
– 단어 시퀀스에 확률을 할당하는 모델, 이전 단어들을 이용하여 다음 단어를 예측함
– BERT는 양쪽 단어들로부터 가운데 단어 예측
- 어떠한 단어가 주어졌을때 다음 단어로 뭐가 나올지 예측하고 확률을 표현

```py
from nltk import bigrams , word_tokenize#영단어 형태소이용,바이그램 이용,.토큰이용
from nltk.util import ngrams #엔그램이용
t= 'I am a boy'#제시
tk  = word_tokenize(t)#토큰화
bg = bigrams(tk)# 직전의 단어
ng = ngrams(tk,3)#특정영역
list(bg), list(ng)
'''
([('I', 'am'), ('am', 'a'), ('a', 'boy')],
 [('I', 'am', 'a'), ('am', 'a', 'boy')])
'''
```
### SS<문장>SE
- 무조건 앞의 정보를 안다고 뒤에올 단어를 맞추느 것도 아니고...
- n그램- 임의의 개수를 정하기위한 기준을 위해 사용한느것,다음예측은 n-1개를 봄
```py
# 왼쪽 오른쪽 공간 추가후 시작단 끝단 표현
data2 = ngrams(tk,2,pad_left=True,pad_right=True
               ,left_pad_symbol="SS", right_pad_symbol="SE")
list(data2)
'''[('SS', 'I'), ('I', 'am'), ('am', 'a'), ('a', 'boy'), ('boy', 'SE')]
'''

# 조건부 확률
from nltk import ConditionalFreqDist
t = 'I am a boy'
t2 = 'you are a boy'
tk = word_tokenize(t)
ng2 = ngrams(tk,2,pad_left=True,pad_right=True
               ,left_pad_symbol="SS", right_pad_symbol="SE")
fd = ConditionalFreqDist([(i[0],i[1]) for i in ng2])
fd.conditions()#정답 접근 정보

from nltk.corpus import movie_reviews
data_l=[]
n=0
for i in movie_reviews.sents():
    bg2 = ngrams(i,2,pad_left=True,pad_right=True
               ,left_pad_symbol="SS", right_pad_symbol="SE")
    data_l += [t for t in bg2]#바이그램 정보
    n+=1

cfd = ConditionalFreqDist(data_l)# 빈도수에 대한 정보를 기;억

cfd["."].most_common(10)#.다음으로 올 단어 빈도 수

from nltk.probability import ConditionalProbDist,MLEProbDist
cpd = ConditionalProbDist(cfd,MLEProbDist)

#특정 단어다음에 목표단어가 올 확률 커스텀

cpd['the'].prob('movie')#the 다음에 movie 가 올 확률
```
```py
#함수사용, 재사용하기위해서

#P(SS<>SE) =P(I|SS)*P(|)*P(|)*P(|) SS뒤에 I가 등장할 확률을 수식으로 나타냄
import numpy as np
def s_sc_f(x):#바이그램 언어 모형의 확률 연산 함수
    p = 0.0 #실수 초기값 설정
    for i in range(len(x)-1): #반복 횟수 설정
        c = x[i]
        w = x[i+1]
        p+= np.log(cpd[c].prob(w) + np.finfo(float).eps)
    return np.exp(p)

test_data=['the','movie','.']#실현 가능 문장
s_sc_f(test_data)
#0.003898785120601922
test_data2=['movie','.','the']#실현 불가능 문장
s_sc_f(test_data2)
#3.085769765203191e-17 (비정상확률)

import random #난수생성
random.seed(10)
cpd['SS'].generate()#랜덤 생성 허나 시드값을 정하면 고정됨
```
- data = ''#str 데이터
- data +=문자열#연결
- 'str'.join()#연결 연산

```py

st = 'SS'
all_str=[]
while True:#무한 반복문
    import random
    random.seed(10)
    st= cpd[st].generate()
    all_str.append(st+' ')
    if st =='SE':
        all_str.pop()
        break
''.join(all_str)
'''"she wasn ' s first part of these guys catch a hard 2 is . "
'''
```
---
## 정리
1. 클래스 생성
2. 데이터 수집
3. 데이터 전처리
    - 토큰화, 정형화, 정규화
4. 모델 학습
5. 모델 학습의 검증(생략)
6. 최종 동작
```py
1.
from nltk.util import ngrams #앞단어와 뒷단어를 갖고있는 data 생성용
from nltk import ConditionalFreqDist #문맥별 단어 빈도수 측정 클래스
from nltk.probability import ConditionalProbDist#조건부 확률 추정 클래스
from nltk.probability import MLEProbDist#최대 우도 추정값을을 도출
# .generate() 샘플 추출 (임의의 값을 추출) 그래서 시드값을 설정함

2.
from nltk.corpus import movie_reviews #모듈을 사용하여 데이터 수집
data = movie_reviews.sents()#단어별 토큰화된 문장들

3.
data_l=[]#처리된 단어 토큰 조합
for i in data:#i는 문장
    bg = ngrams(i,2,pad_left=True,pad_right=True
               ,left_pad_symbol="SS", right_pad_symbol="SE")
    data_l += [t for t in bg]

4. 모델 학습
cfd = ConditionalFreqDist(data_l)#조건부확률처리(토큰리스트),내가입력한 데이터들의 빈도수를 출력 할 수 있다.
cpd = ConditionalProbDist(cfd,MLEProbDist) #단어의 정보를 가짐,그리고 그단어에 대해 샘플들이 존재, = 이빈도수로 추출

5.
st = 'SS'#문장의 시작
all_str=[]
import random
random.seed(10)#seed고정으로 인한 data 고정
while True:#무한 반복문 SS<문장>SE
    st= cpd[st].generate()#임의의 샘플 추출 EX) i - > [am, a, data , SE]
    all_str.append(st)#리스트로 기록, 띄어쓰기 생략
    if st =='SE':#문장의 종료
        all_str.pop()
        break
생성된_data=''.join(all_str)#list의 내용을 이용하여 하나의 문자열로 정리

생성된_data
#'sheandfineeffect;frankly,anddoinghissonbecomesshockinglylazyshortcuttohermotherof"story"'
```
## 미리 수집하여 저장된 데이터 불러와서 사용

```py
#데이터 수집
import codecs
with codecs.open('data2.txt',encoding='utf-8') as f:
    data = [문장.split('\t') for 문장 in f.read().splitlines()]#문장 단위로 
    data = data[1:]
#데이터 전처리
t_data = [문장[1] for 문장 in data]#토큰화
#형태소 토큰화
from konlpy.tag import Okt
tg = Okt()
t=('안녕', 'Noun')
' '.join(t)#튜플 조합
def tk_f(t_data):
    tk_d=['/'.join(x) for x in tg.pos(t_data)]
    return tk_d

from tqdm import tqdm#작업의 진행사항을 알려주는 패키지
from nltk.util import ngrams 
end_data=[]
for i in tqdm(t_data): #반복문을 통한 작업 진행사항 확인
    tk_data = tk_f(i)#형태소 분석기를 이용하여 토큰화
    bg = ngrams(tk_data,2,pad_left=True,pad_right=True
               ,left_pad_symbol="SS", right_pad_symbol="SE")
    end_data +=[t for t in bg]

from nltk import ConditionalFreqDist 
from nltk.probability import ConditionalProbDist,MLEProbDist
cfd = ConditionalFreqDist(end_data)
cpd = ConditionalProbDist(cfd, MLEProbDist)

st='SS'
all_str=[]
import random
random.seed(0)
while True: 
    st=cpd[st].generate()
    if st=='SE':
        break
    d=st.split("/")[0]
    all_str.append(d)
n_data=''.join(all_str)
n_data
#'미키짱과말도전혀빗나가지않던전개로꽥꽥대는거보니까요^^'

def 정리_생성():
    c = "SS"
    sentence = []
    while True:
        w = cpd[c].generate()

        if w == "SE":
            break

        w2 = w.split("/")[0]
        pos = w.split("/")[1]

        if c == "SS":
            sentence.append(w2.title())
        elif c in ["`", "\"", "'", "("]:
            sentence.append(w2)
        elif w2 in ["'", ".", ",", ")", ":", ";", "?"]:
            sentence.append(w2)
        elif pos in ["Josa", "Punctuation", "Suffix"]:
            sentence.append(w2)
        elif w in ["임/Noun", "것/Noun", "는걸/Noun", "릴때/Noun",
                   "되다/Verb", "이다/Verb", "하다/Verb", "이다/Adjective"]:
            sentence.append(w2)
        else:
            sentence.append(" " + w2)
        c = w

    return "".join(sentence)
정리_생성()
''''빨강 머리 박고 싸우는 장면의 호러-> 1 점 만점이다라고 한번은 분명히 보는 편 빼고는 다 스스로 인식에서 폭풍 오열 하는 프로는 장면 속이 시원했어요 주인공이 산만했다 그래도 재밌지 않나? 이 전설의 뜻'
'''
```

## Bag of Words =단어들의 빈도 수
- 단어들의 집합 , 백터화
```py
t = "안녕 나는 강사야 너는 잘 듣고있니?"
#단어간 연관성보단 집합,구분의 목적이라 morphs 로 쪼갬

from konlpy.tag import Okt
tk = Okt()

def 문장_처리(t):
    t= t.replace('?','') #의미없는 단어 정리
    tk_data = tk.morphs(t)#명사로 토큰화
    
    
    idx = {}#단어들의 집합
    d_l=[]#단어 빈도수를 의미
    for i in tk_data:#입력문장
        if i not in idx:#단어가 단어집합key에 있는지?
            idx[i] = len(idx)
            d_l.insert(len(idx)-1,1)#단어 빈도수에 다음값을 삽입
        else:
            ix = idx.get(i)#단어 인덱스는 단어집합
            d_l[ix]+=1
    return idx, d_l
문장_처리(t)

from sklearn.feature_extraction.text import CountVectorizer #BoW
tr_t=t.replace('?','')
#tr_t= tk.morphs(tr_t)

v=CountVectorizer()
v.fit_transform([tr_t]).toarray()
#v.vocabulary_
tr_t#'안녕 나는 강사야 너는 잘 듣고있니'

문장_집합 = ['안녕 나는 강사','나는 학생','나는 공부중','나는 노는중','나는 자는중']

v.vocabulary_# 각 단어들을 인덱싱함
#{'안녕': 4, '나는': 1, '강사야': 0, '너는': 2, '듣고있니': 3}


```

## TF-TDF
– 단어의 빈도와 역 문서 빈도(문서의 빈도에 특정 식을 취함)를 사용하여 DTM 내의 각 단어들마다 중요한 정도를 가중치로 주는 방법
– 주로 문서의 유사도를 구하는 작업, 검색 시스템에서 검색 결과의 중요도를 정하는 작업, 문서 내에서 특정 단어의 중요도를 구하는 작업 등 사용
- tf(d,t) : 특정 문서 d에서의 특정 단어 t의 등장 횟수. 
- df(t) : 특정 단어 t가 등장한 문서의 수.
- idf(d, t) : df(t)에 반비례하는 수.

```py
text=[
    '공부 하고 싶다',
    '공부 하기 싫다',
    '공부 내일 할까',
    '공부 하고 놀기',
    '놀기 말고 놀기'
]
from math import log
import pandas as pd
'''
v=CountVectorizer()
v.fit_transform([tr_t]).toarray()를 간략화
'''
voc = list(set(단어 for 문장 in text for 단어 in 문장.split()))
voc#토큰,리스트화, vocabulary 화라고 봐도됨
#['싶다', '하기', '싫다', '공부', '놀기', '내일', '말고', '할까', '하고'] 
n = len(text)
n # 셋팅된 문장갯수 5개

def tf(t,d):#특정문서 d에서 단어t의 등장횟수
    return d.count(t)
def idf(t):#df(t)에 반비례하는 수
    df=0
    for 문장 in text:
        df += t in 문장
    return log(n/(df+1))
def tfidf(t,d):
    return tf(t,d)*idf(t)

```
## DTM
문서단위 행렬
```py
t_l = []
for i in range(n):
    t_l.append([])
    d=text[i]
    for j in range(len(voc)):
        t = voc[j]
        t_l[-1].append(tf(t,d))
tf_=pd.DataFrame(t_l,columns=voc)
tf_

idf_l=[]
for i in range(len(voc)):
    t=voc[i]
    idf_l.append(idf(t))
idf_=pd.DataFrame(idf_l, index =voc, columns=['IDF'])#시각화
idf_

tf_idf_l =[]

for i in range(n):
    tf_idf_l.append([])
    d=text[i]
    for j in range(len(voc)):
        t = voc[j]
        tf_idf_l[-1].append(tfidf(t,d))
tf_idf_=pd.DataFrame(tf_idf_l,columns=voc)
tf_idf_

```
- 다른방법
```py
#편하게 모듈 사용
from sklearn.feature_extraction.text import TfidfVectorizer
text

tfidfv=TfidfVectorizer().fit(text)# tfidf사용, text는 맨처음에 주어진 데이터
tfidfv.vocabulary_
'''
{'공부': 0,
 '하고': 6,
 '싶다': 5,
 '하기': 7,
 '싫다': 4,
 '내일': 1,
 '할까': 8,
 '놀기': 2,
 '말고': 3}
'''
tfidfv.transform(text).toarray()
#문서단위 행렬이 바로 출력 됨
```