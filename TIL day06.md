## 웹크롤링 복습
 ```python
import requests
from bs4 import BeautifulSoup #정리
url='https://movie.naver.com/movie/point/af/list.naver?&page='#파일 이름
page=10 #몇 페이지인지 입력
r=requests.get(url+str(page)) #겟 메소드 사용
r.raise_for_status()#requests.get쓰면 상태체크해야함,접속상태체크( 접속코드 200이아니면 예외발생)
soup=BeautifulSoup(r.text,"html.parser")#p88 참고
data=soup.find_all("td",attrs={"class":"title"})#td 클래스에서 속성값 타이틀을 찾음
data_l=[]#리스트를 만듬

for i in data:
    if i.a:
        data_l.append({"제목":i.a.text,
        "평점":i.em.text,
        "리뷰":i.br.next_sibling.strip()})#i에 근거하여 리스트에 값 추가
print(data_l)

import sqlite3 #sql쓸려면 필수

conn = sqlite3.connect("data_Ex4.db")#저장할 곳 연결
c = conn.cursor()#커넥션쓰면 필수
c.execute('DROP TABLE IF EXISTS data')
c.execute('''
            CREATE TABLE data(
            제목 text,
            평점 text,
            리뷰 text
             )
            ''')
c.executemany('INSERT INTO data VALUES(:제목,:평점,:리뷰)', data_l)
conn.commit()
conn.close()
db="data_Ex4.db"
def 출력(db):
    conn = sqlite3.connect("data_Ex4.db")
    c = conn.cursor()
    c.execute('SELECT*FROM data')
    for i in c.fetchall():
        print(f"제목:{i[0]},평점:{i[1]},리뷰:{i[2]}")

```
## TIME SLEEP 적용
> 범위 지정 웹스크래핑을 할때 과도한 조회를하면 IP밴의 위험이 있기때문에 사용
```python
import csv #csv 모듈 사용
import time
import requests
from bs4 import BeautifulSoup
url='https://news.naver.com/main/list.naver?mode=LS2D&sid2=228&sid1=105&mid=shm&date=20220404&page='#가져올 링크
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36"}
#requests.get(url,headers=headers)
content="제목","내용"
f=open("save4.csv","w",encoding='utf-8-sig',newline="")
writer=csv.writer(f)
writer.writerow(content)#1차원 데이터 저장
in_data=[]#저장할 공간
#data 수집
for page in range(1,6):
    print(f"page{page} 크롤링중...")
    r=requests.get(url+str(page),headers=headers)
    r.raise_for_status()
    soup=BeautifulSoup(r.text,"html.parser")
    data=soup.find_all("dl")
    #파일 정리
    for i in data:
        if i.a:
            i.dt=i.dt.next_sibling# 추출할 텍스트가 있는 태그 추출
            in_data.append([{"제목": i.dt.next_sibling.text.strip()},
                          {"내용": i.span.text.strip()}])
    time.sleep(5)#5초의 딜래이
#저장
writer.writerows(in_data)#2차원 에 추출

```