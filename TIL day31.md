# NLP 를 위한 딥러닝 복습
[책링크](https://wikidocs.net/book/2155)
1. 단어 토큰화
```py
t= "Time is an illusion. Lunchtime double so!"

from nltk.tokenize import word_tokenize
word_tokenize(t)
-> ["Time", "is", "an", "illustion", "Lunchtime", "double", "so"]

from tensorflow.keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(t)
-> ['time', 'is', 'an', 'illusion', 'lunchtime', 'double', 'so']
```
2. 문장 토큰화
```py
from nltk.tokenize import sent_tokenize

text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print('문장 토큰화1 :',sent_tokenize(text))

'''문장 토큰화1 : ['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to make sure no one was near.']
'''

```
3. 한국어 토큰화
- 한국어는 띄어쓰기 기준으로 토큰화를 잘 안한다. 대신에 형태소 기준으로 토큰화를 한다.
- 형태소 분석기를 사용하는 것. konlpy, khaiii, mecab

```py
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
tokenized_sentence = word_tokenize(text)

print('단어 토큰화 :',tokenized_sentence)
print('품사 태깅 :',pos_tag(tokenized_sentence))
'''
단어 토큰화 : ['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
품사 태깅 : [('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]
'''
영어는 이렇고 한국어는
from konlpy.tag import Okt
from konlpy.tag import Kkma
from konlpy.tag import Hannanum

okt = Okt()
kkma = Kkma()
han = Hannanum()

print('OKT 형태소 분석 :',okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 품사 태깅 :',okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 명사 추출 :',okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요")) 

'''
OKT 형태소 분석 : ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']
OKT 품사 태깅 : [('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]
OKT 명사 추출 : ['코딩', '당신', '연휴', '여행']'''

print('꼬꼬마 형태소 분석 :',kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('꼬꼬마 품사 태깅 :',kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('꼬꼬마 명사 추출 :',kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요")) 
'''
꼬꼬마 형태소 분석 : ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']
꼬꼬마 품사 태깅 : [('열심히', 'MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]
꼬꼬마 명사 추출 : ['코딩', '당신', '연휴', '여행']
'''
print('한나눔 형태소 분석 :',han.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('한나눔 품사 태깅 :',han.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('한나눔 명사 추출 :',han.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요")) 

'''
한나눔 형태소 분석 : ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에는', '여행', '을', '가', '아', '보', '아']
한나눔 품사 태깅 : [('열심히', 'M'), ('코딩', 'N'), ('하', 'X'), ('ㄴ', 'E'), ('당신', 'N'), (',', 'S'), ('연휴', 'N'), ('에는', 'J'), ('여행', 'N'), ('을', 'J'), ('가', 'P'), ('아', 'E'), ('보', 'P'), ('아', 'E')]
한나눔 명사 추출 : ['코딩', '당신', '연휴', '여행']
'''

```
```py
import kss

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
print('한국어 문장 토큰화 :',kss.split_sentences(text))
'''
한국어 문장 토큰화 : ['딥 러닝 자연어 처리가 재미있기는 합니다.', '그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다.', '이제 해보면 알걸요?']
'''

```
4. 정수 인코딩
```py
from tensorflow.keras.preprocessing.text import Tokenizer

preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

tokenizer = Tokenizer() #토크나이저 불러옴
tokenizer.fit_on_texts(preprocessed_sentences) #학습
print(tokenizer.word_index) #학습 확인하기
tokenizer.texts_to_sequences(preprocessed_sentences) #정수인코딩


```
5. 패딩
```py
from tensorflow.keras.preprocessing.sequence import pad_sequences
encoded = tokenizer.texts_to_sequences(preprocessed_sentences)# 정수 인코딩

padded = pad_sequences(encoded)#패딩화
padded

padded = pad_sequences(encoded, padding='post')#뒤로 0을 채우고 싶으시다면
padded

padded = pad_sequences(encoded, padding='post', maxlen=5)#최대길이를 정하고 싶으시다면
padded
```

6. 원 핫 인코딩
- 보통 라벨 값을 원핫인코딩을 하지만 여기선 예시로..
```py
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"
sub_text = "점심 먹으러 갈래 메뉴는 햄버거 최고야"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
encoded = tokenizer.texts_to_sequences([sub_text])[0]
print(encoded)
#[2, 5, 1, 6, 3, 7]

one_hot = to_categorical(encoded)
print(one_hot)
'''
[[0. 0. 1. 0. 0. 0. 0. 0.] # 인덱스 2의 원-핫 벡터
 [0. 0. 0. 0. 0. 1. 0. 0.] # 인덱스 5의 원-핫 벡터
 [0. 1. 0. 0. 0. 0. 0. 0.] # 인덱스 1의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 1. 0.] # 인덱스 6의 원-핫 벡터
 [0. 0. 0. 1. 0. 0. 0. 0.] # 인덱스 3의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 0. 1.]] # 인덱스 7의 원-핫 벡터
'''
```

7. 머신러닝
```py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

x = np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) # 숫자 10부터 1

model = Sequential()
model.add(Dense(1, input_dim=1, activation='sigmoid'))

sgd = optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(x, y, epochs=200)
```
```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_test, y_test))

```

8. 딥러닝

```py
mport pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer#토큰
from tensorflow.keras.utils import to_categorical#원핫인코딩(정수형 범주)에만
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
import numpy as np
#데이터 로드
X_data= pd.read_csv('X_data.csv')
Y_data=pd.read_csv('Y_data.csv')#보캐뷸러데이터임

X_data.shape,Y_data.shape
#((10000, 10000), (10000, 103))
#학습데이터 분할
t_x,tt_x,t_y,tt_y=train_test_split(X_data,Y_data)

# MLP 다중뉴럴네트워크
#피드 포워드 신경망, 데이터가 영향을안줌
m=Sequential()
m.add(Dense(256,input_shape=(10000,),activation='relu'))
m.add(Dropout(0.5))
m.add(Dense(128,activation='relu'))
m.add(Dropout(0.5))
m.add(Dense(103,activation='softmax'))#다중 데이터를 처리하지만, 확률을 도출
m.compile(optimizer='adam',loss='categorical_crossentropy',
          metrics=['accuracy'])
hy=m.fit(t_x,t_y,epochs=10,validation_data=(tt_x,tt_y))

ec=range(1,len(hy.history['accuracy'])+1)#그래프 길이를 반복횟수만큼..
plt.plot(ec,hy.history['loss'])
plt.plot(ec,hy.history['val_loss'])

```

9. RNN

```py
from tensorflow.keras.layers import SimpleRNN

model.add(SimpleRNN(hidden_units)) 
# 추가 인자를 사용할 때
model.add(SimpleRNN(hidden_units, input_shape=(timesteps, input_dim)))

# 다른 표기
model.add(SimpleRNN(hidden_units, input_length=M, input_dim=N))
```