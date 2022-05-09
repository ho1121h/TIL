# 텍스트 분석 2
1. 공백 처리
```py 
#1. 모듈없이 공백제거
t='난 공백이 있어 왜 있냐? 바이는 바보 주먹밖에 몰라'#원본
nt = t.replace(' ','')

#2.공백 제거된것을 다시 띄우기
from pykospacing import Spacing #형태소에 맞게 띄어쓰기를 함
sc = Spacing()
e_t = sc(nt)
#음,,, 띄어쓰기가 원본처럼 안됨
#형태소파악 모듈
from hanspell import spell_checker
ck_t = spell_checker.check(t1)
end_t = ck_t.checked#형태소 파악 하고 판단한다
ck_t1 = spell_checker.check(t).checked#이제 원본처럼 띄어쓰기가 처리됨
```
2. 형태소 분석 평가
```py
from soynlp import DoublespaceLineCorpus #자연어 처리 가능
from soynlp.word import WordExtractor#형태소를 분석하기 위한 클래스

all_data = DoublespaceLineCorpus('data1.txt')#문서 단위 토큰 생성
len(all_data)#문장크기 = 30091
w_e = WordExtractor()
w_e.train(all_data)#학습작업
w_e_t = w_e.extract()#단어에 대한것을 토큰화/규칙을 설정

w_e_t['반포한강공원'].cohesion_forward#특정단어를 말하다. =그단어가 모여질 확률 0.378
#자립 형태소+의존 형태소
w_e_t['반포한강공원에'].cohesion_forward#응집 확률 0.33

w_e_t['반포한강공원'].right_branching_entropy#해당 키의 밸류값의 점수를 추출

#자립형태소와 의존형태소를 쪼개다. / 텍스트-문장화-단어화-형태소
n=0
for i in w_e_t.items():
    print(i)
    n+=1
    if n==3:
        break
#Scores 의 점수로 형태소를 파악
'''
'텟', Scores(cohesion_forward=0
'끌', Scores(cohesion_forward=0
'이', Scores(cohesion_forward=0
'''
from soynlp.tokenizer import LTokenizer #L토큰화 작업,왜 전달하나? 
#기존 점수로 형태소를 분류를 하는데 락습된모델에서 추출하는데 점수기준으로
sc = {w:sc.cohesion_forward for w, sc in w_e_t.items()}#키의 밸류를 다음 형태소점수(cohesion_forward)로 재설정
n=0
for i in sc.items():
    print(i)
    n+=1
    if n==3:
        break

l_tk=LTokenizer(scores=sc)
ck_t='자료의 정보를 구분 하기 위해 문서 작성을 했다.'
l_tk.tokenize(ck_t)#제대로 분류가 되진않았다
'''['자료', '의', '정보를', '구분', '하기', '위해', '문서', '작성을', '했다', '.']
'''
from soynlp.tokenizer import MaxScoreTokenizer#맥스 스코어 ,얼마나 유사한지~
m_tk = MaxScoreTokenizer(scores=sc)
ck_t='자료의 정보를 구분 하기 위해 문서 작성을 했다.'
m_tk.tokenize(ck_t)
'''['자료', '의', '정보를', '구분', '하기', '위해', '문서', '작성을', '했다', '.']
'''
```

3. data 를 불러와서 문장토큰화 처리

```py 
#data를 문장으로 가져오기위해
from soynlp import DoublespaceLineCorpus #자연어 처리 가능
from soynlp.word import WordExtractor#형태소를 분석하기 위한
all_data = DoublespaceLineCorpus('data1.txt', iter_sent=True)
len(all_data)#223357 문장으로 꺼내서 ex1의 데이터보다 더많아짐

n = 0
for i in all_data:
    print(i)
    n+=1
    if n == 4:
        break

m=WordExtractor()
m.train(all_data)#학습데이터
m.extract()#해당 (단어)데이터 유사성으로 구축

#중복처리
t1 = '영화가 너무 웃겨'
t2 = '영화가 너무 웃겨 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ'
t3 = '영화가 너무 웃겨 ㅋㅋ'
t4 = '영화가 너무 웃겨 ㅋㅋㅋㅋ'

from soynlp.normalizer import *
emoticon_normalize(t1)#본 nlp로 데이터를 정리하고, 중복되는 단어를 줄임
emoticon_normalize(t2,num_repeats=1)#반복되는 단어의 반복되는 횟수=매개변수설정
t_e="후후후후후훗"
emoticon_normalize(t_e)
''' '후후훗'
'''
```

4. 형태소, 단어 태그를 커스텀
- 명사가 아님-> 명사
```py
from konlpy.tag import Okt#한국어 분석기
from ckonlpy.tag import Twitter #한국어 분석기를 보조 원하는 것을 태그

t = '은호가 교실로 들어갔다.'# 명사처리 할 단어 제시

#3가지의 형태소 분류기
tw = Okt()
tw.nouns(t)#['은', '호가', '교실']
#tw.nouns#명사 추출
#tw.morphs#형태소 추출
#tw.pos#품수 추출
tw.morphs(t)#['은', '호가', '교실', '로', '들어갔다', '.']
tw.pos(t)#품사 태그
'''
[('은', 'Noun'),
 ('호가', 'Noun'),
 ('교실', 'Noun'),
 ('로', 'Josa'),
 ('들어갔다', 'Verb'),
 ('.', 'Punctuation')]
'''
from konlpy.tag import Twitter
tw1 = Twitter()
tw1.morphs(t)
#['은', '호가', '교실', '로', '들어갔다', '.'] 
from ckonlpy.tag import Twitter#트위터 클래스를 보조함
tw2 =Twitter()
tw2.morphs(t)#['은', '호가', '교실', '로', '들어갔다', '.']

#보조해주는 장치에 add로 추가할것,단어와 품사를 추가하다
tw2.add_dictionary('은호','Noun')#은호를 명사로 추가
tw2.morphs(t)#['은호', '가', '교실', '로', '들어갔다', '.']
#명사로 처리됨을 볼수 있다.
tw2.add_dictionary('은이','Noun')
t2='은호가 교실로 은이가 집으로 들어갔다.'
tw2.pos(t2)

tw3 = Twitter()
tw3.pos(t2)#내가 커스텀한 객체에만 명사가 적용됨을 알 수 있다.
#은호와 은이가 명사처리가 안됬다 tw3에 add로 처리안해서 그럼 ㅇㅇ

tw2.morphs(t2)#tw2(학습이끝난객체)토큰화가 끝남 
'''['은호', '가', '교실', '로', '은이', '가', '집', '으로', '들어갔다', '.']
'''
```
---

5. 저장된 문서 정보 불러오기, 형태소 분석기로 토큰화하기,정형화 하기

```py
from konlpy.corpus import kolaw #문서정보 드로우
kolaw.fileids()
#로컬 저장소에 저장된 ['constitution.txt'] 추출
t_data = kolaw.open('constitution.txt').read()
from konlpy.corpus import kobill
kobill.fileids()
#로컬저장소에 저장된 텍스트 목록 불러오기
t_data2 = kobill.open('1809890.txt').read() #해당정보 선택
t_data2#대한민국 헌법이 어쩌구 저쩌구 정보가 나옴


from konlpy.tag import Okt#형태소 분석기를 이용하여 단어 토큰화
tw1 = Okt()
s_data = tw1.nouns(t_data2)#명사만 가져오기
s_data=[i for i in s_data if len(i)>1] #명사 추출,리스트 구조
s_data# 2글자 이상 명사만! 추출


```
만들어진 데이터를 단어 리스트를 이용하여 단어의 정형화의 결과를 출력하시오
- 토큰화,정형화,정수화
```py
from tensorflow.keras.preprocessing.text import Tokenizer
tk = Tokenizer()
tk.fit_on_texts(s_data)
encoded = tk.texts_to_sequences(s_data)
tk.index_word #단어갯수별 나열


#문서 로드->문장토큰->단어토큰->정수 인코딩- >패딩
from konlpy.corpus import kolaw
from tensorflow.keras.preprocessing.sequence import pad_sequences
t_data = kolaw.open('constitution.txt')
from soynlp import DoublespaceLineCorpus
t_data = DoublespaceLineCorpus(t_data.name, iter_sent=True)#문장 토큰화 3번에서 배운것

#형태분석기 이용
tw=Okt()
d=[]
for i in t_data:
    s_data=tw.nouns(i)#명사추출
    s_data=[i for i in s_data if len(i)>1]#단어 1개초과 재정의
    d.append(s_data)
tk2 = Tokenizer()#단어 토큰화
tk2.fit_on_texts(d)# 처리완료된 토큰 학습

encoded3 = tk2.texts_to_sequences(d)#정수 인코딩
tk2.index_word#키,밸류 목록을 볼 수있다.

#정수인코딩한걸 패딩화
end_data2 = pad_sequences(encoded3,padding='post',truncating='post')#패딩 결과
end_data2

---

#단어 토큰화
tk2 = Tokenizer()
tk2.fit_on_texts(t_data)
encoded2 = tk2.texts_to_sequences(t_data)#정수 인코딩
tk2.index_word

# 토큰을 명사화 1개이상 처리 안하고 패딩화
end_data = pad_sequences(encoded2,padding='post')#패딩2
end_data
```

6. 단어를 불러온 다음 워드클라우드화
```py
from konlpy.corpus import kolaw
kolaw.fileids()
t_data=kolaw.open('constitution.txt').read()
from konlpy.tag import Okt #형태소 분석기
tw = Okt()
n_t_data = tw.nouns(t_data)#명사를 추출
n_t_data #단어 토큰화, 형태소 분류됨, 토큰화 된 데이터들은 리스트가됨
tw.tagset
'''
{'Adjective': '형용사',
 'Adverb': '부사',
 'Alpha': '알파벳',
 'Conjunction': '접속사',
 'Determiner': '관형사',
 'Eomi': '어미',
 'Exclamation': '감탄사',
 'Foreign': '외국어, 한자 및 기타기호',
 'Hashtag': '트위터 해쉬태그',
 'Josa': '조사',
 'KoreanParticle': '(ex: ㅋㅋ)',
 'Noun': '명사',
 'Number': '숫자',
 'PreEomi': '선어말어미',
 'Punctuation': '구두점',
 'ScreenName': '트위터 아이디',
 'Suffix': '접미사',
 'Unknown': '미등록어',
 'Verb': '동사'}
'''

from nltk import Text#단어취급 클래스
ck_data=Text(n_t_data,name='kolaw')
ck_data

import matplotlib.pyplot as plt
plt.rc('font', family='NanumGothic')
ck_data.plot(15)
plt.show()#빈도 수 순으로 앞에서 15개를 표로 나타냄

ck_data.vocab()#빈도수 표현 이거랑 보면 이해됨

#워드크라우드화
from wordcloud import WordCloud#한글을 플롯에 나타내기위해 워드클라우드 사용
f=r'C:\Users\student\AppData\Local\Microsoft\Windows\Fonts\NanumGothic.ttf'
wc=WordCloud(font_path=f,width=1000,height=600,background_color='white')
#워드클라우드 표현
plt.imshow(wc.generate_from_frequencies(ck_data.vocab()))#빈도수 표현을 자료로 사용
plt.axis('off')
```


7. ***대충 정리***
```py
1. data 수집

from konlpy.corpus import kolaw
t_data = kolaw.open('constitution.txt').read()
type(t_data)

2. 데이터 전처리- 형태소 분석기 이용


from konlpy.tag import Okt #형태소 분석기
tw = Okt()
n_t_data = tw.nouns(t_data)#명사를 추출
type(n_t_data) #단어 토큰화, 형태소 분류됨

3.시각화 작업을 위한 Text nltk 클래스이용 정리

from nltk import Text#단어취급 클래스
ck_data = Text(n_t_data, name='kolaw')
ck_data

4. 정리된 내용을 이용하여 data 분석

plt.axis('off')
from wordcloud import WordCloud#한글을 플롯에 나타내기위해 워드클라우드 사용
f = r'C:/Users/ho316/AppData/Local/Microsoft/Windows/Fonts/Maplestory Light.ttf'
wc = WordCloud(font_path=f,width=1000, height=600,background_color='white')
plt.imshow(wc.generate_from_frequencies(ck_data.vocab()))
plt.axis('off')

```

## 인코딩, 벡터화
- 학습을 한뒤 단어 토큰화
- DictVectorizer => BOW 인코딩 벡터
```py
from sklearn.feature_extraction import DictVectorizer

v = DictVectorizer(sparse=False)
#특정단어가 얼마나 수치적으로..
D=[{'A':1,'B':2},{'B':3,'C':1}]#3종-> A가 0인덱스 B가 1인덱스 C가 2인덱스
X = v.fit_transform(D)#학습!
X
'''
array([[1., 2., 0.],   첫 리스트는 A가1개 B가 2개 C가 0개
       [0., 3., 1.]])   두번째 리스트는 A가 0개 B가 3개 C가 1개
'''
v.feature_names_ #학습된 피쳐이름 ['A', 'B', 'C']

D2 = [{'A':5, 'B':1,'D':100}]
v.transform(D2)#앞에서 학습한것에 D가 없어요~
'''
array([[5., 1., 0.]])
'''
```
- CountVectorizer => BOW 인코딩 벡터 문서의 집합 정보를 단어토큰화 생성-> 각 단어의 수를 셈
```py
from sklearn.feature_extraction.text import CountVectorizer
#문서 집합의 정보(토큰)
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
    'The last document?',
]

v1 = CountVectorizer()
v1.fit(corpus)#학습
v1.vocabulary_#10칸의 인덱스 생성,벡터화
#주어진 문장토큰을 스스로 학습해서 인덱스 칸을 나눔
'''
{'this': 9,
 'is': 3,
 'the': 7,
 'first': 2,
 'document': 1,
 'second': 6,
 'and': 0,
 'third': 8,
 'one': 5,
 'last': 4}
'''
v1.transform(['This is the first document. This This']).toarray()
#위에서 단어별 인덱스에 몇개가 있는지 출력됨
'''
array([[0, 1, 1, 1, 0, 0, 0, 1, 0, 3]], dtype=int64)
'''
v1.transform(['This is the first document.data ']).toarray()
#위에서 학습한 인덱싱리스트에 없는 단어 data는 카운팅이 되지 않음
```