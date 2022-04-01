## hrml 문자열
- match("문자열"):처음부터 일치
- search("문자열"):일치하는 문자 있는지 확인
- findall("문자열"):일치 하는 모든것의 리스트 출력
- . 문자 ("a.a")
- ^ 시작("^a")
- $ 끝 ("a$")


## 웹 크롤링
- 크롤링은 웹 페이지를 그대로 가져와 데이터를 추출해 내는 행위
- 크롤링하는 소프트웨어는 크롤러

--- 

## Beautifulsoup 
- 뷰디풀수프는 html을 수프객체로 만들어 추출하기 쉽게함

- 예제
 ```python
import requests
from bs4 import BeautifulSoup

url ='크롤링할 사이트'
import requests
r=requests.get(url)
soup=BeautifulSoup(r.text,"html.parser")

   ```