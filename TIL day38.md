# 웹 크롤링 - 리퀘스트
```
웹 스크레이핑
스크레이핑이란 웹크롤링이라고도함
웹사이트에서 원하는 정보를 자동으로 수집하고 정리하는 것
```
- TCP 응답이있는 통신 
- UDP 응답이없는 통신

## 리퀘스트

```py
import requests 
#requests.<http_method>(url )

resp = requests.get("https://naver.com")
resp # 200 = 응답 성공

resp.headers #해더

resp.content[:1000] #바디

response = requests.get("https://search.naver.com/search.naver", params={'where':'news', 'query':'무역전쟁'})
response 

response.content[:1000]

response.text

sample = resp.json() #제이슨으로 불러오기
sample.keys()
sample['data']
```
# XML Parsing
- 뷰티풀 수프
```py
import requests
from bs4 import BeautifulSoup # 클래스는 공백으로 구분

resp = requests.get("https://www.naver.com/")
soup = BeautifulSoup(resp.text)

group_nav_tag =soup.find('div', class_ ='group_nav')
type(group_nav_tag)

group_nav_tags = soup.find_all('div', class_ = 'group_nav') # 파인드 올로 전부 가져옴
type(group_nav_tags)

a_tags = group_nav_tag.find_all('a') # a 태그를 전부 가져옴
a_tags
'''
[<a class="nav" data-clk="svc.mail" href="https://mail.naver.com/"><i class="ico_mail"></i>메일</a>,
 <a class="nav" data-clk="svc.cafe" href="https://section.cafe.naver.com/">카페</a>,
 <a class="nav" data-clk="svc.blog" href="https://section.blog.naver.com/">블로그</a>,
'''
sample = a_tags[0]
sample

result = [{'url': tag.get('href'),'name':tag.text} for tag in a_tags]
result
'''
[{'url': 'https://mail.naver.com/', 'name': '메일'},
 {'url': 'https://section.cafe.naver.com/', 'name': '카페'},
 {'url': 'https://section.blog.naver.com/', 'name': '블로그'},
'''
# 위방법과 같지만 어려우면 아래걸 사용
result_set = []
for tag in a_tags:
    url = tag.get('href')
    name = tag.text
    result_set.append({
        'url':url,
        'name':name
    })
print(result_set)


data = soup.select('a.nav') # css 셀렉터 사용 , 클래스는 .사용
data

data = soup.select('div#NM_FAVORITE') # id 는 #을 사용
data

data = soup.select('div.group_nav li.nav_item') # ~안의 ~을 가져오기
data
```
- 1번 문제 : 기사 제목, 기사 url, 기사 내용압축 크롤링

```py
import pandas as pd

result = [{'name':tag.find('a',class_ = 'news_tit').text, 'url': tag.find('a',class_ = 'news_tit').get('href'), 'desc':tag.find('a',class_ = 'api_txt_lines dsc_txt_wrap').text} for tag in div_tag]
df1 =pd.DataFrame(result)
df1


```
- 2 번 문제: 10페이지까지 크롤링


```py
result_set = []
for i in range(10):
    resp = requests.get(f'https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%EC%9A%B0%ED%81%AC%EB%9D%BC%EC%9D%B4%EB%82%98%20%EB%9F%AC%EC%8B%9C%EC%95%84&sort=0&photo=0&field=0&pd=0&ds=&de=&cluster_rank=61&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:all,a:all&start={i}1')
    
    soup = BeautifulSoup(resp.text)
    div_tag =soup.select('div.news_area')
    for tag in div_tag:
        name = tag.find('a',class_ = 'news_tit').text
        url = tag.find('a',class_ = 'news_tit').get('href')
        desc = tag.find('a',class_ = 'api_txt_lines dsc_txt_wrap').text
        result_set.append({
            'name':name,
            'url':url,
            'desc':desc
        })
print(result_set)
df2 = pd.DataFrame(result_set)
df2


```

- 정답

```py
# 정답

base_url = 'https://search.naver.com/search.naver'
base_params = {
    'where':'news',
    'query':"우크라이나 러시아",
    'start':1
}
result_list = []
for i in range(10):
    base_params['start'] = i*10+1
    resp = requests.get(url, base_params)
    
    soup = BeautifulSoup(resp.text)
    new_list_tags = soup.select('.list_news > .bx')   
    
    for new_list_tag in new_list_tags:
        title_tag = new_list_tag.selesct_one('.news_tit')
        title = title_tag.text
        url =title_tag.get('href')
        desc = news_list_tag.select_one('.dsc_txt_wrap').text
        result_list.append({
            'title':title,
            'url':url,
            'desc':desc
        })
result_list


```

## 와디즈 크롤링 - JSON + 파라미터 설정

1. 와디즈 페이지 접속
2. 소스 검사 
3. 네트워크에서 AJAX 문서 찾기
4. 페이로드 분석해서 파라미터 값 찾기
<br>tNum: 528
limit: 48
order: recommend
keyword: 
endYn: all <br>
로 되어있을 것

```PY
import requests
from tqdm import tqdm



params = dict(
    starNum =0,
    limit = 100,
    order ='recommend',
    keyword='',
    endYn='ALL',
    )

data = []
for i in tqdm(range(10)):
    params.update({'starNum':i*100, 'limit':100})
    resp = requests.get('https://www.wadiz.kr/web/wreward/ajaxGetCardList',params=params)
    data.extend(resp.json()['data'])

import pandas as pd
df = pd.DataFrame(data)
df.head()

```

## 당근마켓 크롤링 연습
```py
import requests
from bs4 import BeautifulSoup
webpage  = requests.get("https://www.daangn.com/hot_articles")
# 예시로 중고거래 인기 매물
soup = BeautifulSoup(webpage.content , "html.parser")
getitem = soup.select("#content > section.cards-wrap > article")#css셀렉터

for item in getitem: # 전체 값에서 하나씩 꺼내오닌깐 0인덱스로 꺼내와야 오류가 없을것
#제목
 print( item.select('a > div.card-desc > h2 ')[0].text.strip(), end=",")
#가격
 print( item.select('a > div.card-desc > div.card-price')[0].text.strip(),end=",")
#위치
 print( item.select('a > div.card-desc > div.card-region-name')[0].text.strip(),end=",")
#관심,채팅( 인기도)
 print( item.select('a > div.card-desc > div.card-counts')[0].text.strip().replace(' ', '').replace('\n', '').replace('∙', ','))

# 데이터에 담기 

from tqdm import tqdm

item_list = []

for item in tqdm(getitem):
    #제목
         a = item.select('a > div.card-desc > h2 ')[0].text.strip()
    #가격
         b= item.select('a > div.card-desc > div.card-price')[0].text.strip()
    #위치
         c =item.select('a > div.card-desc > div.card-region-name')[0].text.strip()
    #관심,채팅( 인기도)
         d= item.select('a > div.card-desc > div.card-counts')[0].text.strip().replace(' ', '').replace('\n', '').replace('∙', ',')
        
         item_list.append({
             '제목' : a,
             '가격' : b,
             '위치' : c,
             '관심,채팅' : d
         })
import pandas as pd

df = pd.DataFrame(item_list)
```