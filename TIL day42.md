## 문자열 문제  1
-단어 공부 

```py
# 알파벳 대소문자로 된 단어가 주어지면, 이 단어에서 가장 많이 사용된 알파벳이 무엇인지 알아내는 프로그램을 작성하시오. 단, 대문자와 소문자를 구분하지 않는다.

# 링크 : https://www.acmicpc.net/problem/1157

import sys
input = sys.stdin.readline

word = input().strip() # 개행 문자땜시 스트립 추가

word = word.lower()
# print(list(set(word)))
word2 = list(set(word))

# max를 쓰면 간단히 나오지만 max가 여러개일 경우인 경우 ?를 출력하기위해 사용
cnt = []
for i in word2 :
    # print(i)
    count = word.count(i)
    # print(count)
    cnt.append(count)
# print(cnt.count(max(cnt)))

if cnt.count(max(cnt)) >=2 : # max(cnt) 가 2개 이상이라면 ? 출력
    print("?")

else: # cnt 값이 가장 큰 문자를 set된 리스트에서 반환 하고 대문자로 출력
    print(word2[(cnt.index(max(cnt)))].upper())

```
- 단순히 입력받고 lower , upper 한뒤 max(문자열) 하면 해결되지않는가? 하고 생각했으나 결과는 중복 값때문에 안됨

- 결론은 set으로 정렬한 뒤 인덱스로 제일 큰값을 반환하는걸로 풀면 됨