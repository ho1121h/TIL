# 통계 기반 비정형 텍스트
## 데이터수집
– 데이터 전처리
    - 자연어 처리에서 크롤링 등으로 얻어낸 코퍼스 데이터가 필요에 맞게 전처리되지 않은 상태시 해당 데이터를 용도에 맞게 토큰화(tokenization) & 정제(cleaning) & 정규화(normalization)를 진행 해야 함
1. 토큰화
    - 단어 토큰화
        - 토큰의 기준을 단어(word)로 하는것
    - 문장 토큰화
2. 정제
    - 갖고 있는 코퍼스로부터 노이즈 데이터 제거
    - 완벽한 정제는 어렵고 합의점을 찾아야함
3. 정규화
    - 표현 방법이 다른 단어들을 통합시켜서 같은 단어로 만듬
    - 규칙에 기반한통합, 대소문자 통합, 불필요한 단어 제거
- 정규화 기법
    - 어간추출: 어간 stem을 추출
    - 표제어 추출
---

1. 단어 토큰화
- 구두점이나 특수 문자를 단순 제외하면 안됨
- 아포스트로피나 줄임말 문제
```py
import nltk ,kss , konlpy
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence#워드 변환
tt1="Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
word_tokenize(tt1) #단어토큰화 방법 1

WordPunctTokenizer().tokenize(tt1)#(대문자)클래스에 인스턴스로 접근 (방법2)->don't 가 인식됨

text_to_word_sequence(tt1)#방법3


from nltk.tokenize import TreebankWordTokenizer# - 를 묶어서 출력하기ㅇㅇㅇㅇㅇ
t2="Don't be fooled by the dark sounding name, home-data Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
t2
TreebankWordTokenizer().tokenize(t2)
word_tokenize(t2)#단어 토큰화
```

2. 문장 토큰화
```py
t3="His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
from nltk.tokenize import sent_tokenize #문장 토큰화
sent_tokenize(t3)
'''
['His barber kept his word.',
 'But keeping such a huge secret to himself was driving him crazy.',
 'Finally, the barber went up a mountain and almost to the edge of a cliff.',
 'He dug a hole in the midst of some reeds.',
 'He looked about, to make sure no one was near.']
'''
t4="His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, a Ph.D. the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
sent_tokenize(t4)
'''
['His barber kept his word.',
 'But keeping such a huge secret to himself was driving him crazy.',
 'Finally, a Ph.D. the barber went up a mountain and almost to the edge of a cliff.',
 'He dug a hole in the midst of some reeds.',
 'He looked about, to make sure no one was near.']
'''

```
3. 정제
- **형태소**: 뜻을가진 가장 작은 말의 단위 . 한국어 토큰화에서는 형태소란 개념을 반드시 이해 해야한다. 
    - 자립 형태소
    - 의존 형태소
- 에디가 책을 읽었다-> 에디, 책 / 가 ,을, 읽, 었, 다
- 띄어쓰기에 따라 단어뜻이 달라지는데 한국어는 띄어쓰기를 안해도 뜻이 전달되기에 형태소를 이해해야한다
[형태소](https://konlpy.org/ko/latest/morph/#pos-tagging-with-konlpy)
```py
t6="I am actively looking for Ph.D. students. and you are a Ph.D. student."
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
t_t = word_tokenize(t6)
pos_tag(t_t)#태깅

#한글은 형태소에 태그를 붙인다
from konlpy.tag import Okt
from konlpy.tag import Kkma
n1 = Okt()
#형태소
t7="우리는 즐거운 어린이날 부터 휴일 입니다. 고생한 여러분 휴일을 즐기세요."
n1.morphs(t7)

n1.pos(t7)#태깅해주는 메소드
'''
[('우리', 'Noun'),
 ('는', 'Josa'),
 ('즐거운', 'Adjective'),
 ('어린이날', 'Noun'),
 ('부터', 'Noun'),
 ('휴일', 'Noun'),
 ('입니다', 'Adjective'),
 ('.', 'Punctuation'),
 ('고생', 'Noun'),
 ('한', 'Josa'),
 ('여러분', 'Noun'),
 ('휴일', 'Noun'),
 ('을', 'Josa'),
 ('즐기세요', 'Verb'),
 ('.', 'Punctuation')]
'''
n1.nouns(t7)#명사추출
```

- 어간(stem)

```py
words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
from nltk.stem import WordNetLemmatizer
f=WordNetLemmatizer().lemmatize

#표제어 추출(근원추출)
[f(x) for x in words]
f('has','v') #have
f('dies','v')#die

t1 = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
from nltk.tokenize import word_tokenize #토큰화
data = word_tokenize(t1)
data
from nltk.stem import PorterStemmer #어간추출 클래스
f1 = PorterStemmer().stem
[f1(x) for x in data]#리스트화

from nltk.stem import PorterStemmer #포터 알고리즘 어간 추출 클래스
from nltk.stem import LancasterStemmer
#어간추출 = 데이터가 단어 상태일때 사용가능
f2 = PorterStemmer().stem
f3 = LancasterStemmer().stem
d1 = [f2(x) for x in words]
d2 = [f3(x) for x in words]
print(words, d1 ,d2, sep='\n') #어떤 알고리즘을 사용 했는가에 따라 다름!
#과도한 정제화는 필요없다 원래 뜻이 변형됨
'''
['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
['polici', 'do', 'organ', 'have', 'go', 'love', 'live', 'fli', 'die', 'watch', 'ha', 'start']
['policy', 'doing', 'org', 'hav', 'going', 'lov', 'liv', 'fly', 'die', 'watch', 'has', 'start']
'''
```
- 불용어
```py
#불용
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt #형태소
d=word_tokenize(t1)

ck_d = stopwords.words('english')#불용 속성

end_l=[]
for i in d:
    if i not in ck_d:
        end_l.append(i)
print(d,end_l,sep='\n')#불용어 제거 처리
'''
d = ['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
end_l = ['I', 'actively', 'looking', 'Ph.D.', 'students', '.', 'Ph.D.', 'student', '.']
'''
#불용어 직접 지정방법
t_d='오늘은 몸이 아파요. 오늘은 강의 내용이 빠르게 진행이 되나요 모르겠어요'
stop_t_d = '은 요 빠르게 모르겠어요'.split()
okt = Okt()
t_dd = okt.morphs(t_d)#형태소로 토큰화
end_d = [i for i in t_dd if not i in stop_t_d]#불용어와 겹치는 단어가 다제거됨
```
4. 정규화
- 정규표현식 참고

```py
import re
# . 기호-(단어 1개)
r =re.compile('d.t') #하나의 문자
r.search('adatadaasdsadadsatd') #span=(1, 4)

# ? 기호 - 임력 이후 중간 문자의 유무 상관없이 끝문자가 나오면 인식
r =re.compile('dsa?t') 
r.search('adsasasasdsat')# span=(9, 13)

# *기호 - 연결된문자의 갯수 상관없이 인식, 0개부터 시작가능
r=re.compile('ab*c')
r.search('ac')#span=(0, 2)
r.search('abbbbbbbbc')#span=(0, 10)

# +기호 - 연결된문자의 갯수 상관없이 인식 갯수1개부터 가능
r = re.compile('ab+c')
r.search('abc')#span=(0, 3)

# ^기호 - 기호 이후 문자열로 시작되는 문자열 인식
r = re.compile('^ab')
r.search('abcssssscccsdsadav')#span=(0, 2)

#{숫자}기호 - 기호 앞에 있는 단어의 갯수가 기호 안의 숫자만큼 있을 때 인식
r = re.compile('ab{2}c')
r.search('abbc')#span=(0, 4)

#{숫자,숫자}기호 - 기호 앞에 있는 단어의 갯수가 기호 안의 숫자 범위 만큼 있을 때 인식.
r = re.compile('ab{2,5}c')
r.search('abbbbbc')#span=(0, 7)

#{숫자,}기호-- 슬라이싱이라고 생각하면됨
r = re.compile('ab{2,}c')
r.search('abbbbbbbbbbbbbbbbc')#span=(0, 18)

#[]기호- 괄호 안에 있는 문자의 유무, [-]
r = re.compile('[abc]')
r.search('eeeccc')#괄호안에있으면 첫만남을 인식
r = re.compile('[a-z]')
r.search('y')#a~z 알파벳 사이에 포함되면 바로 인식

#정규식 함수
r = re.compile('a.c')#조건 설정
r.search('data0- aac')#만족하기만 하면 인식
#<re.Match object; span=(7, 10), match='aac'>
r.match('aac')# 시작부터 확인 후 인식,<re.Match object; span=(0, 3), match='aac'>
data='data1 data2 data3'
re.split(' ',data)#자르기 (조건, data),['data1', 'data2', 'data3']
# \w는 [a~z,A~Z,0~9] 문자또는 숫자 의미
# \W는 문자 또는 숫자가 아닌 문자를 의미
re.findall('\w+',data)
re.sub('\W+',"수정",data)#공백(문자아님)에 ""을 대입 'data1수정data2수정data3'

#함수로 토큰화
t="Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
from nltk.tokenize import RegexpTokenizer
ck1 = RegexpTokenizer('[\w]+')
ck1.tokenize(t)
# \s는 공백을 의미
ck2 = RegexpTokenizer('\s+',gaps=True)
ck2.tokenize(t)
```

- 정수인코딩

```py
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# data 수집
text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."
# 데이터 전처리 =문장 토큰화
문장_토큰화 = sent_tokenize(text)
문장_토큰화#문장 토큰화됨 ㅇㅇ

#데이터 전처리 = 단어 토큰화 but 정제화 정규화를 해야함
단어_모음={}#키밸류
pr_data=[]
불용성단어 = set(stopwords.words('english'))

for 문장 in 문장_토큰화:#문장 토큰화 된 토큰들에서 꺼냄
    단어_토큰화 = word_tokenize(문장)#단어 토큰화
    l= []
    for 단어 in 단어_토큰화:
        소문자화_된_단어 = 단어.lower()#소문자화
        if 소문자화_된_단어 not in 불용성단어:
            if len(소문자화_된_단어)>2:#단어수 2이하 제거
                l.append(소문자화_된_단어)
                if 소문자화_된_단어 not in 단어_모음:
                    단어_모음[소문자화_된_단어]=0
                단어_모음[소문자화_된_단어]+=1
    pr_data.append(l)
pr_data#단어 토큰화****
'''
[['barber', 'person'],
 ['barber', 'good', 'person'],
 ['barber', 'huge', 'person'],
 ['knew', 'secret'],
 ['secret', 'kept', 'huge', 'secret'],
 ['huge', 'secret'],
 ['barber', 'kept', 'word'],
 ['barber', 'kept', 'word'],
 ['barber', 'kept', 'secret'],
 ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'],
 ['barber', 'went', 'huge', 'mountain']]
'''

n=4          # 키       키 ,밸류           키밸류에서    밸류가 4초과
삭제_결정된_data = [단어 for 단어,i in 단어_인덱스.items() if i>n]
for i in 삭제_결정된_data:#초과된 데이터 삭제
    del 단어_인덱스[i]
단어_인덱스

단어_인덱스['OOV']= len(단어_인덱스)+1
단어_인덱스        
'''{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'OOV': 5}'''
# 문자열을 정수 화 
ec_data = []
for i in pr_data:
    ec_d=[]
    for 단어 in i:
        try:#예외발생시 except 블록으로 이동
            ec_d.append(단어_인덱스[단어])
        except KeyError: #해당에러 발생시
            ec_d.append(단어_인덱스['OOV'])#없는단어집합에 추가
    ec_data.append(ec_d)
ec_data #정수화 
'''
[[1, 5],
 [1, 5, 5],
 [1, 3, 5],
 [5, 2],
 [2, 4, 3, 2],
 [3, 2],
 [1, 4, 5],
 [1, 4, 5],
 [1, 4, 2],
 [5, 5, 3, 2, 5, 1, 5],
 [1, 5, 3, 5]]

'''

```
- Counter 기반 정수인코딩

```py
from collections import Counter
pr_data#텍스트->문장->단어 토큰화
단어_모음집 = sum(pr_data,[])
단어_모음집#대괄호 없어짐 1차원화
결과_단어_모음집 = Counter(단어_모음집)
결과_단어_모음집
n = 4
빈도수별_단어 = 결과_단어_모음집.most_common(n)
빈도수별_단어
단어_인덱스2={}
i = 0
for 단어,빈도수 in 빈도수별_단어:
    i+=1
    단어_인덱스2[단어]=i
단어_인덱스2
단어_인덱스2['OOV'] = len(단어_인덱스2)+1
단어_인덱스2
#정수화
ec_data2 = []
for i in pr_data:
    ec_d2=[]
    for 단어 in i:
        try:#예외발생시 except 블록으로 이동
            ec_d2.append(단어_인덱스2[단어])
        except KeyError: #해당에러 발생시
            ec_d2.append(단어_인덱스2['OOV'])#없는단어집합에 추가
    ec_data2.append(ec_d2)
ec_data2 #정수화 
```
- NLTk정수 인코딩
```py
from nltk import FreqDist
import numpy as np
pr_data

단어_모음 = FreqDist(np.hstack(pr_data))
단어_모음#단어별 키밸류

단어_모음1 = 단어_모음.most_common(4)
단어_모음1#[('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4)]

from tensorflow.keras.preprocessing.text import Tokenizer

ck_t = Tokenizer()
ck_t.fit_on_texts(pr_data)
ck_t.word_index#빈도 등수 오름차순
ck_t.word_counts#빈도 수

ck_t.texts_to_sequences(pr_data)#정수화
```
```py
from tensorflow.keras.preprocessing.text import Tokenizer
tk = Tokenizer()
tk.fit_on_texts(pr_data)
encoded = tk.texts_to_sequences(pr_data)
encoded
'''
[[1, 5],
 [1, 8, 5],
 [1, 3, 5],
 [9, 2],
 [2, 4, 3, 2],
 [3, 2],
 [1, 4, 6],
 [1, 4, 6],
 [1, 4, 2],
 [7, 7, 3, 2, 10, 1, 11],
 [1, 12, 3, 13]]
'''
max_l = max(len(x)for x in encoded)#제일 긴 리스트 = 7

#max_l 로 패딩
import numpy as np
for i in encoded:
    while len(i)<max_l:
        i.append(0)
data= np.array(encoded)
data
'''
array([[ 1,  5,  0,  0,  0,  0,  0],
       [ 1,  8,  5,  0,  0,  0,  0],
       [ 1,  3,  5,  0,  0,  0,  0],
       [ 9,  2,  0,  0,  0,  0,  0],
       [ 2,  4,  3,  2,  0,  0,  0],
       [ 3,  2,  0,  0,  0,  0,  0],
       [ 1,  4,  6,  0,  0,  0,  0],
       [ 1,  4,  6,  0,  0,  0,  0],
       [ 1,  4,  2,  0,  0,  0,  0],
       [ 7,  7,  3,  2, 10,  1, 11],
       [ 1, 12,  3, 13,  0,  0,  0]])
'''
# 다른방법으로 패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
tk = Tokenizer()
tk.fit_on_texts(pr_data)
encoded = tk.texts_to_sequences(pr_data)
encoded
end_data = pad_sequences(encoded,padding='post')#패딩
end_data

#리스트 사이즈 조절 패딩
end_data = pad_sequences(encoded,padding='post',truncating='post',maxlen=5)#패딩,데이터 사이즈
end_data
'''
array([[ 1,  5,  0,  0,  0],
       [ 1,  8,  5,  0,  0],
       [ 1,  3,  5,  0,  0],
       [ 9,  2,  0,  0,  0],
       [ 2,  4,  3,  2,  0],
       [ 3,  2,  0,  0,  0],
       [ 1,  4,  6,  0,  0],
       [ 1,  4,  6,  0,  0],
       [ 1,  4,  2,  0,  0],
       [ 7,  7,  3,  2, 10],
       [ 1, 12,  3, 13,  0]])
'''
#제일 긴데이터가 13
#제일긴 리스트길이7칸
v = len(tk.word_index)+1
end_data = pad_sequences(encoded,padding='post',truncating='post',value=v)#빈공간에 v값 넣음
end_data
'''
array([[ 1,  5, 14, 14, 14, 14, 14],
       [ 1,  8,  5, 14, 14, 14, 14],
       [ 1,  3,  5, 14, 14, 14, 14],
       [ 9,  2, 14, 14, 14, 14, 14],
       [ 2,  4,  3,  2, 14, 14, 14],
       [ 3,  2, 14, 14, 14, 14, 14],
       [ 1,  4,  6, 14, 14, 14, 14],
       [ 1,  4,  6, 14, 14, 14, 14],
       [ 1,  4,  2, 14, 14, 14, 14],
       [ 7,  7,  3,  2, 10,  1, 11],
       [ 1, 12,  3, 13, 14, 14, 14]])
'''

```
- 결론: 정수 인코딩은 단어를 숫자로 표현하되 빈도수가 높을수록 작은값으로 표현
- 원핫 인코딩
    - 단어 집합의 크기를 벡터의 차원으로 하고 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 벡터 표현 방식 dummies
- 단어집합
    - 기본적으로 북과 북스와 같이 단어의 변형 형태도 다른 단어로 간주
- 원핫의 한계
    - 벡터를 저장하기위해 공간이 점점 늘어남 ->차원늘어남->과대적합