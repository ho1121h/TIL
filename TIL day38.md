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


## 번개장터 크롤링 연습
- 리퀘스트 
- 번개장터 사이트에서 소스코드 보기 한 후 json 주소 찾기
-  검색할 키워드와 크롤링할 페이지 수 를 input 한다
- 
```py

pid = [] # 제품 아이디
category = input()
for page in tqdm(range(int(input()))):
    url = f'https://api.bunjang.co.kr/api/1/find_v2.json?order=date&n=96&page={page}&req_ref=search&q={category}&stat_device=w&stat_category_required=1&version=4'
    response = requests.get(url)
    datas = response.json()['list']
    
    ids = [data['pid'] for data in datas] #리스트 생성
    pid.extend(ids)# 추가
items=[]
for i in pid: # 제품 아이디 하나씩 훑기
    url = f'https://api.bunjang.co.kr/api/1/product/{i}/detail_info.json?version=4'
    response = requests.get(url) # 리퀘스트
    try:
        details = response.json()['item_info'] #제이슨 파일을 그대로 가져오되 카테고리 이름, 페이옵션을 삭제
        details.pop('category_name')
        details.pop('pay_option')
        items.append(details)# 가져오면 딕셔너리 형태임
    except:# 오류나면 에러 출력
        print('error')

#데이터프레임화
df = pd.DataFrame(items)
bunjang_df = df[['name','price','location','description_for_detail','num_item_view','pid']]
bunjang_df = bunjang_df.rename({'name':'title','location':'region','description_for_detail':'desc','num_item_view':'view_counts'},axis='columns')
bunjang_df['link'] = 'https://m.bunjang.co.kr/products/'+ bunjang_df['pid']
bunjang_df['market'] = '번개장터'
bunjang_df['keyword'] = category
bunjang_df.drop(['pid'], axis=1)

# 전처리
bunjang_df.reset_index(drop = True)
bunjang_df['desc'] = bunjang_df['desc'] \
.replace(r'[^가-힣 ]', ' ', regex=True) \
.replace("'", '') \
.replace(r'\s+', ' ', regex=True) \
.str.strip() \
.str[:255]
bunjang_df = bunjang_df[bunjang_df['desc'].str.strip().astype(bool)]

bunjang_df.to_csv("번개장터크롤링.csv",encoding="utf-8")
```
## 타임스탬프 변환
- 유닉스 시계를 기준으로 측정함
```py
from datetime import datetime
# 일-월-년 으로 출력할려면 사용


bunjang_df['update_time']=bunjang_df['update_time'].map(lambda x : datetime.fromtimestamp(x).strftime('%y-%m-%d') ) 
bunjang_df['update_time']


```
- 추가 전처리
```py
#bunjang_df.query('desc.str.contains("매입|삽니다|구매|최고가|전기종")', engine='python')

drop_list = bunjang_df.query('desc.str.contains("매입|삽니다|구매|최고가|전기종")', engine='python').index
drop_bunjang_df = bunjang_df.drop(drop_list)


drop_bunjang_df.price = drop_bunjang_df.price.astype('float')

q1= drop_bunjang_df['price'].quantile(0.25)
q3= drop_bunjang_df['price'].quantile(0.75)
condition=drop_bunjang_df['price']>q3+1.5*iqr

drop_price_list = drop_bunjang_df[condition].index

drop_bunjang_df.drop(drop_price_list, inplace= True)

drop_bunjang_df.reset_index(inplace= True)

drop_bunjang_df.drop('index', axis= 1, inplace= True)

drop_bunjang_df
```

## 당근마켓 크롤링
```py
import requests
import json
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

item_list = []
category =input() # 검색어 입력

for i in tqdm(range(1,int(input()))):
    url =f"https://www.daangn.com/search/{category}/more/flea_market?page={i}"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content , "html.parser")
    getitem = soup.select(".flea-market-article")
    
    for item in getitem:   
        try:
            item_list.append({
                 'title' : item.select('.article-title')[0].text.strip(),
                 'price' : item.select('.article-price')[0].text.strip(),
                 'region' : item.select('.article-region-name')[0].text.strip(),
                 'desc' : item.select('.article-content')[0].text.strip()
             })
        except:
            print('error')

import pandas as pd

df = pd.DataFrame(item_list)
df.head()
df.reset_index(drop = True)
df['desc'] = df['desc'] \
.replace("'", '') \
.replace(r'\s+', ' ', regex=True) \
.str.strip() \
.str[:255]
df = df[df['desc'].str.strip().astype(bool)]

df['price'] = df['price'].str.replace("만",'0000')
df['price'] = df['price'].str.replace(r'[^0-9]', '', regex=True)
df['price']=df['price'].replace('^ +','')
df['price']=df['price'].replace('',np.nan) 
df=df.dropna(how='any') 
df
df.price = df.price.astype('float')

q1= df['price'].quantile(0.25)
q3= df['price'].quantile(0.75)
iqr=q3-q1
condition=df['price']>q3+1.5*iqr

drop_price_list = df[condition].index

df.drop(drop_price_list, inplace= True)

df.reset_index(inplace= True)

df.drop('index', axis= 1, inplace= True)

df
df.price = df.price.astype('float')

q1= df['price'].quantile(0.25)
q3= df['price'].quantile(0.75)
iqr=q3-q1
condition=df['price']>q3+1.5*iqr

drop_price_list = df[condition].index

df.drop(drop_price_list, inplace= True)

df.reset_index(inplace= True)

df.drop('index', axis= 1, inplace= True)

df




```