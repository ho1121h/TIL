# 웹 크롤링 복습 2
```python
#1.네이버 뉴스에 암호화폐 검색후 뉴스 제목과 내용 5p 스크래핑하기
#2.csv 파일로 저장
#3.저장된 파일을 이용하여 출력
import csv
from selenium import webdriver
from bs4 import BeautifulSoup
browser = webdriver.Chrome()
browser.implicitly_wait(10)
browser.maximize_window()
url = "https://naver.com"
browser.get(url)
browser.find_element_by_xpath('//*[@id="query"]').send_keys('암호화폐\n')
browser.implicitly_wait(10)
browser.find_element_by_xpath('//*[@id="lnb"]/div[1]/div/ul/li[2]/a').click()
title="제목","내용"
f=open("Q2.csv","w",encoding='utf-8-sig',newline="")
writer=csv.writer(f)
writer.writerow(title)
l=[]
for i in range(1,6):
    print(f"{i}페이지")
    browser.implicitly_wait(10)
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    browser.implicitly_wait(10)
    html=browser.page_source
    browser.implicitly_wait(10)
    browser.find_element_by_xpath(f'//*[@id="main_pack"]/div[2]/div/div/a[{i + 1}]').click()
    soup = BeautifulSoup(html, "html.parser")
    data=soup.select('div.news_area')
    for i in data:
        if i.a:
            l.append([i.select_one('a.news_tit').text,i.select_one('a.api_txt_lines.dsc_txt_wrap').text])
    print("화면전환")
writer.writerows(l)
f.close()
f=open("Q2.csv","r",encoding='utf-8-sig',newline="")
reader=csv.reader(f)
for i in reader:
    print(i[0],i[1])#리더로 리스트의 값을 읽어옴

```
## headless 를 이용
- 헤들리스를 이용하면 웹창(테스트창)없이 실행 가능하다.
```python
#*headless를 이용하여 웹창없이 실행
#1.네이버 뉴스에 {자유검색} 검색후 뉴스 제목과 내용 5p 스크래핑하기
#2.csv 파일로 저장
#3.저장된 파일을 이용하여 출력
import csv
from selenium import webdriver
from bs4 import BeautifulSoup
op=webdriver.ChromeOptions()
op.headless = True
op.add_argument("window-size=1920x1080")
op.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36')
b=webdriver.Chrome(options=op)#크롬의 옵션사용
b.maximize_window()#최대창 크기로 사용
url = "https://naver.com"
b.get(url)
b.find_element_by_xpath('//*[@id="query"]').send_keys('암호화폐\n')
b.implicitly_wait(10)
b.find_element_by_xpath('//*[@id="lnb"]/div[1]/div/ul/li[2]/a').click()
title="제목","내용"
f=open("Q3.csv","w",encoding='utf-8-sig',newline="")
writer=csv.writer(f)
writer.writerow(title)
l=[]
for i in range(1,6):
    print(f"{i}페이지")
    b.implicitly_wait(10)
    b.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    b.implicitly_wait(10)
    html=b.page_source
    b.implicitly_wait(10)
    b.find_element_by_xpath(f'//*[@id="main_pack"]/div[2]/div/div/a[{i + 1}]').click()
    soup = BeautifulSoup(html, "html.parser")
    data=soup.select('div.news_area')
    for i in data:
        if i.a:
            l.append([i.select_one('a.news_tit').text,i.select_one('a.api_txt_lines.dsc_txt_wrap').text])#리스트에 추가할 원소
    print("화면전환")
writer.writerows(l)#l이라는 리스트에 저장
f.close()# 닫기
f=open("Q3.csv","r",encoding='utf-8-sig',newline="")
reader=csv.reader(f)#csv에 저장된 정보 모두 불러오기
for i in reader:
    print(i[0],i[1])#개별 원소 출력

```