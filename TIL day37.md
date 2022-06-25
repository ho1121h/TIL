# 유용한 라이브러리

- 최대 공약수와 최대 공배수 구하기
```py
import math

def lcm(a, b):
    return a * b // math.gcd(a, b)

a = 21
b = 14

print(math ,gcd(21, 14)) #최대 공약수
print(lcm(21, 14)) # 최소 공배수
```

- 순열과 조합
<br>조합: 서로 다른 n개에서 순서에 상관 없이 서로 다른 r개를 선택하는 것<br>

```py
from itertools import combinations
data = ['A', 'B', 'C']

result = list(combinations(data, 2))
print(result)

```
<br>순열: 서로 다른 n개에서 서로다른 r개를 선택하여 일렬로 나열하는 것 <br>

```py
from itertools import permutations
data = ['A', 'B', 'C']

result = list(permutations(data, 3))
print(result)

```


- 중복 순열과 중복 조합


```py 
from itertools import product #중복순열

data = ['A', 'B', 'C']

result = list(product(data, repeat = 2))
print(result)

from itertools import combinations_with_replacement #중복 조합

data = ['A', 'B', 'C']

result = list(combinations_with_replacement(data, 2))
print(result)
```
