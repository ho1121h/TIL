# BERT 버트 모델
- 사전 훈련된 워드 임베딩
    - 동음 이의어의 문제를 해결할 수 있다.


## 마스크드 언어 모델
1. transformers 패키지를 사용하여 모델과 토크나이저를 로드합니다.
```py
'''
TFBertForMaskedLM.from_pretrained('BERT 모델 이름')을 넣으면 [MASK]라고 되어있는 단어를 맞추기 위한 마스크드 언어 모델링을 위한 구조로 BERT를 로드합니다. 다시 말해서 BERT를 마스크드 언어 모델 형태로 로드합니다.

AutoTokenizer.from_pretrained('모델 이름')을 넣으면 해당 모델이 학습되었을 당시에 사용되었던 토크나이저를 로드합니다.
'''
from transformers import TFBertForMaskedLM
from transformers import AutoTokenizer

model = TFBertForMaskedLM.from_pretrained('bert-large-uncased')
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
# 학습시킬때 전부 소문자로 
#해당 문자를 정수인코딩
inputs = tokenizer('Soccer is a really fun [MASK].', return_tensors='tf')
print(inputs['input_ids'])
#tf.Tensor([[ 101 4715 2003 1037 2428 4569  103 1012  102]], shape=(1, 9), dtype=int32)

print(inputs['token_type_ids'])#세그먼트 인코딩 결과(첫문장)
#tf.Tensor([[0 0 0 0 0 0 0 0 0]], shape=(1, 9), dtype=int32)

```
- 마스크 토큰 예측
```py
from transformers import FillMaskPipeline
pip = FillMaskPipeline(model=model, tokenizer=tokenizer)

pip('Soccer is a really fun [MASK].')# 상위 5개 출력
'''
[{'score': 0.7621126174926758,
  'token': 4368,
  'token_str': 'sport',
  'sequence': 'soccer is a really fun sport.'},
 {'score': 0.2034195512533188,
  'token': 2208,
  'token_str': 'game',
  'sequence': 'soccer is a really fun game.'},
 {'score': 0.012208537198603153,
  'token': 2518,
  'token_str': 'thing',
  'sequence': 'soccer is a really fun thing.'},
 {'score': 0.0018630226841196418,
  'token': 4023,
  'token_str': 'activity',
  'sequence': 'soccer is a really fun activity.'},
 {'score': 0.0013354863040149212,
  'token': 2492,
  'token_str': 'field',
  'sequence': 'soccer is a really fun field.'}]
'''

```

## 한국어 버트
- 다음 문장 예측 모델과 토크나이저
이제 TFBertForNextSentencePrediction를 통해서 다음 문장을 예측해봅시다. 
모델에 입력을 넣으면, 해당 모델은 소프트맥스 함수를 지나기 전의 값인 logits을 리턴합니다. 
해당 값을 소프트맥스 함수를 통과시킨 후 두 개의 값 중 더 큰 값을 모델의 예측값으로 판단하도록 더 큰 확률값을 가진 인덱스를 리턴하도록 합니다.

```py
import tensorflow as tf
from transformers import TFBertForNextSentencePrediction# 다음 문장 예측 모듈
from transformers import AutoTokenizer
import torch

model = TFBertForNextSentencePrediction.from_pretrained('klue/bert-base', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
```
```py
# 이어지는 두 개의 문장
prompt = "2002년 월드컵 축구대회는 일본과 공동으로 개최되었던 세계적인 큰 잔치입니다."
next_sentence = "여행을 가보니 한국의 2002년 월드컵 축구대회의 준비는 완벽했습니다."
encoding = tokenizer(prompt, next_sentence, return_tensors='tf')

logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]

softmax = tf.keras.layers.Softmax()
probs = softmax(logits)
print('최종 예측 레이블 :', tf.math.argmax(probs, axis=-1).numpy())
#최종 예측 레이블 : [0]

```
실질적으로 서로 이어지는 두개의 문장 레이블은 0이된다.

```py
# 상관없는 두 개의 문장
prompt = "2002년 월드컵 축구대회는 일본과 공동으로 개최되었던 세계적인 큰 잔치입니다."
next_sentence = "극장가서 로맨스 영화를 보고싶어요"
encoding = tokenizer(prompt, next_sentence, return_tensors='tf')

logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]

softmax = tf.keras.layers.Softmax()
probs = softmax(logits)
print('최종 예측 레이블 :', tf.math.argmax(probs, axis=-1).numpy())
#최종 예측 레이블 : [1]
```
이어지지 않는 문장의 레이블은 1이된다. 왜냐면 두개의 조건으로 이진분류로 학습을 하였기 때문이다.