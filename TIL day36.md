# 키워드 추출 
- 요약이 하고싶은가?
```py 
# 준비물
import numpy as np
import itertools

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
doc = """
         Supervised learning is the machine learning task of 
         learning a function that maps an input to an output based 
         on example input-output pairs.[1] It infers a function 
         from labeled training data consisting of a set of 
         training examples.[2] In supervised learning, each 
         example is a pair consisting of an input object 
         (typically a vector) and a desired output value (also 
         called the supervisory signal). A supervised learning 
         algorithm analyzes the training data and produces an 
         inferred function, which can be used for mapping new 
         examples. An optimal scenario will allow for the algorithm 
         to correctly determine the class labels for unseen 
         instances. This requires the learning algorithm to  
         generalize from the training data to unseen situations 
         in a 'reasonable' way (see inductive bias).
      """
#문서준비


```
- 사이킷런의 카운터 벡터를 사용하여 단어를 추출한다. 그 이유는 엔그램의 인자를 사용하면 쉽게 엔그램을 추출 할 수 있기 때문이다. (3,3)으로 설정하면 결과 후보는 3개의 단어를 한 묶음으로 간주하는 트리그램을 추출함
```py
# 3개의 단어 묶음인 단어구 추출
n_gram_range = (3, 3)
stop_words = "english"

count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc]) # 문서로 학습
candidates = count.get_feature_names()

print('trigram 개수 :',len(candidates))# 72 개
print('trigram 다섯개만 출력 :',candidates[:5])# 

```
- 이제 문서와 문서로분터 추출한 키워드로 버트를 통해서 수치화 한다.
```py
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)

```
- 문서와 가장 유사한 키워드들을 추출한다.

```py
top_n = 5
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
print(keywords)
#['algorithm analyzes training', 'learning algorithm generalize', 'learning machine learning', 'learning algorithm analyzes', 'algorithm generalize training']
```
- 5개의 키워드가 출력 됬으나, 이들의 의미가 비슷해보임. 이키워드들이 문서를 가장 잘나타내기 때문, 키워드들을 다양하게 출력하고 싶으면 다음을 사용

- 다양성을 위해 두가지 알고리즘을 사용
    - Max Sum Similarity
    - Maximal Marginal Relevance

## Max Sum Similarity
데이터 쌍 사이의 최대 합 거리는 데이터 쌍 간의 거리가 최대화되는 데이터 쌍으로 정의됨 
여기서 의도는 후보간의 유사성을 최소화하면서 문서와 후보 유사성을 극대화 하고자 한다.
```py
def max_sum_sim(doc_embedding, candidate_embeddings, words, top_n, nr_candidates):
    # 문서와 각 키워드들 간의 유사도
    distances = cosine_similarity(doc_embedding, candidate_embeddings)

    # 각 키워드들 간의 유사도
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # 코사인 유사도에 기반하여 키워드들 중 상위 top_n개의 단어를 pick.
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # 각 키워드들 중에서 가장 덜 유사한 키워드들간의 조합을 계산
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]
'''
이를 위해 상위 10개의 키워드를 선택하고 이 10개 중에서 서로 가장 유사성이 낮은 5개를 선택합니다. 낮은 nr_candidates를 설정하면 결과는 출력된 키워드 5개는 기존의 코사인 유사도만 사용한 것과 매우 유사한 것으로 보입니다.
'''
# 함수 사용(키버트를 이용해수치화한문서,키버트를 이용해 수치화한 단어 , 키워드, 갯수)
max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=10)
'''
['requires learning algorithm',
 'signal supervised learning',
 'learning function maps',
 'algorithm analyzes training',
 'learning machine learning']
'''
max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=50) # 더욱 다양한 키워드
'''
['pairs infers function',
 'used mapping new',
 'algorithm correctly determine',
 'training data consisting',
 'learning machine learning']
'''
```
## Maximal Marginal Relevance
결과를 다양히 하기위해 텍스트 요약 작업에서 중복을 최소화하고 결과의 다양성을 극대화 하기위해 노력함 mmr은 문서와 가장 유사한 키워드/키프레이즈를 선택함. 그런다음 문서와 유사하고 이미선택된 키워드/키프레이즈와 유사하지 않은 새로운 후보를 반복적으로 선택한다.

```py
def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

    # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    # 각 키워드들 간의 유사도
    word_similarity = cosine_similarity(candidate_embeddings)

    # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # keywords_idx = [2]
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

```
 만약 상대적으로 낮은 diversity 값을 설정한다면, 결과는 기존의 코사인 유사도만 사용한 것과 매우 유사한 것으로 보임

```py
# 함수(수치화된 문서, 수치화된 단어, 단어)
mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.2)
'''
['algorithm generalize training',
 'supervised learning algorithm',
 'learning machine learning',
 'learning algorithm analyzes',
 'learning algorithm generalize']
'''
mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.7)# 더 다양함
'''
['algorithm generalize training',
 'labels unseen instances',
 'new examples optimal',
 'determine class labels',
 'supervised learning algorithm']
'''
```