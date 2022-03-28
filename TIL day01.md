# TIL day 01
# 제목(Heading)

## 헤딩 2

### 헤딩3

#### 헤딩 4

##### 헤딩 5

###### 헤딩 6

---

목록

순서가 없느 목록

- 목록 1
- 목록 2
- 목록 3
  - 서브 목록1
    - 서브 서브 목록
      - 서브 서브 서브 목록
- 목록 4
- 

---

강조

'**볼드체(굵)**'

'*이탤릭체(기울임)*'

'~~취소선~~'

---

소스 코드

1. 인라인

   한 줄의 코드를 쓸 수 있도록 하는 코드 블럭

   `print("Hello World")`

2. 블록

   여러 줄의 코드를 쓸 수 있도록 하는 코드 블럭

   ```python
   for i in range(10):
       print(i)	
   ```

---
삽입 기능

링크 `[링크](링크주소)`

이미지 `[이미지이름]URL`

주석

목차 '삽입후 제목에따라 분류됨'

HTML
---
## git hub 사용하기 
### git에게 작성자 누구인지 알려주기
git config --global user.name 이름
git config --global user.email 이메일주소

git config --global --list

### 일반 폴더 -> 로컬 저장소
**git init**

### 상태확인
**git status**

### Working Directory -> Staging Area
**git add a.txt** 'a.txt는 파일이름'

### Staging Area -> Commits
**git commit** -m "first commit"

### commit 확인(버전확인)
**git log** 여러줄로 버전확인
**git log --oneline** 한줄로 버전확인

### git 을 허브에 업로드
*** git push origin master ***