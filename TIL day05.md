## hrml 문자열
- match("문자열"):처음부터 일치
- search("문자열"):일치하는 문자 있는지 확인
- findall("문자열"):일치 하는 모든것의 리스트 출력
- [re 정규식](https://docs.python.org/ko/3/library/re.html)
- . 문자 ("a.a")
- ^ 시작("^a")
- $ 끝 ("a$")
--- 

## 웹 크롤링
- 크롤링은 웹 페이지를 그대로 가져와 데이터를 추출해 내는 행위
- 크롤링하는 소프트웨어는 크롤러

## html.parser
- 간단한 HTML과 XHTML 구문 분석기
> HTMLParser 인스턴스는 HTML 데이터를 받아서 시작 태그, 종료 태그, 텍스트, 주석 및 기타 마크업 요소를 만날 때마다 처리기 메서드를 호출
- [HTMLParser 메서드](https://docs.python.org/ko/3/library/html.parser.html?highlight=html#module-html.parser)

## Beautifulsoup 
- 단순한 api가 특징인 스크래핑 라이브러리
- 뷰디풀수프는 html을 수프객체로 만들어 추출하기 쉽게함
- 우선 크롬에 접속 후 F12를 누르고 html 속성을 살펴보자
- 예제
 ```python
import requests
from bs4 import BeautifulSoup
url ='크롤링할 사이트'
import requests
r=requests.get(url)
soup=BeautifulSoup(r.text,"html.parser")
data=soup.select("찾을속성")
for i in data:
    print(i.text)
```

```python
import requests
from bs4 import BeautifulSoup
url ='크롤링할 사이트'
import requests
r=requests.get(url)
soup=BeautifulSoup(r.text,"html.parser")
for i in soup.select('td[class=left]'):
    if i.a:
        print(i.a.text)
```