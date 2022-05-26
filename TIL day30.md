# 웹 크롤링 복습 2~3
- 인스타 사진을 크롤링 해보자
```py 
import pandas as pd
import numpy as np

# 라이브러리 import
# 라이브러리 : 필요한 도구
from selenium import webdriver  # 라이브러리(모듈) 가져오라
from selenium.webdriver import ActionChains as AC
import chromedriver_autoinstaller
from tqdm import tqdm
from tqdm import tqdm_notebook
import re
from time import sleep
import time

# 워닝 무시
import warnings
warnings.filterwarnings('ignore')
# 데이터 수집할 키워드 지정
keyword = "검색할 키워드"
len_insta = 20  # 몇 개의 게시글을 수집할 것인가

# url에 검색 'keyword' 입력
url = "https://www.instagram.com/explore/tags/{}/".format(keyword)

chrome_path = chromedriver_autoinstaller.install()
driver = webdriver.Chrome(chrome_path)
driver.get(url)

# 수동 로그인
# id, pw 입력
from getpass import getpass

user_id = getpass("사용자 id를 입력하시오.")#님아이디
user_pw = getpass("사용자 pw를 입력하시오.")#님 패스워드

# 로그인 버튼 클릭
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# 자동으로 로그인 페이지로 가지 않았다면 이동
if not driver.current_url.startswith("https://www.instagram.com/accounts/login/"):
    css_selector = "a.ENC4C:nth-child(1)"
    driver.find_element(By.CSS_SELECTOR, css_selector).click()
    time.sleep(3)

# id 입력
css_selector = "div.-MzZI:nth-child(1) > div:nth-child(1) > label:nth-child(1) > input:nth-child(2)"
driver.find_element(By.CSS_SELECTOR, css_selector).send_keys(user_id)
driver.find_element(By.CSS_SELECTOR, css_selector).send_keys(Keys.TAB)
time.sleep(2)

# pw 입력
css_selector = "div.-MzZI:nth-child(2) > div:nth-child(1) > label:nth-child(1) > input:nth-child(2)"
driver.find_element(By.CSS_SELECTOR, css_selector).send_keys(user_pw)
driver.find_element(By.CSS_SELECTOR, css_selector).send_keys(Keys.ENTER)
time.sleep(5)

# 만약 id 저장할꺼냐고 물어보면 no
css_selector = "button.sqdOP:nth-child(1)"
try:
    driver.find_element(By.CSS_SELECTOR, css_selector).click()
except:
    pass
time.sleep(10)

dict = {}  # 전체 게시글을 담을 그릇

# 첫번째 사진 클릭 ,클래스명 =eLAPa
CSS_tran=".eLAPa"
elements = driver.find_elements_by_css_selector(CSS_tran)   # 사진 클릭
elements[0].click()
time.sleep(2)

# 크롤링 시작
for j in tqdm_notebook(range(0, len_insta)):    

    target_info = {}                                            # 사진별 데이터를 담을 딕셔너리 생성

    try:    # 크롤링을 시도해라. 오류가나면 건너띄기
        # 사진(pic) 크롤링
        overlays1 = ".ZyFrc .FFVAD"                   # 사진창 속 사진   
        img = driver.find_element_by_css_selector(overlays1)    # 사진 선택
        pic = img.get_attribute('src')                          # 사진 url 크롤링 완료
        target_info['picture'] = pic

        # 날짜(date) 크롤링
        overlays2 = "._1o9PC"                # 날짜 지정
        datum2 = driver.find_element_by_css_selector(overlays2)     # 날짜 선택
        date = datum2.get_attribute('title')
        target_info['date'] = date

        # 좋아요(like) 크롤링
        overlays3 = "._7UhW9.xLCgt.qyrsm.KV-D4.fDxYl.T0kll"                                        # 리뷰창 속 날짜
        datum3 = driver.find_element_by_css_selector(overlays3)     # 리뷰 선택
        like = datum3.text                                          # 좋아요 크롤링 완료
        target_info['like'] = like

        # 해시태그(tag) 크롤링
        overlays4 = ".xil3i"                                         
        datum3 = driver.find_elements_by_css_selector(overlays4)    # 태그 선택
        tag_list = []
        for i in range(len(datum3)):
            tag_list.append(datum3[i].text)
        target_info['tag'] = tag_list

        dict[j] = target_info            # 토탈 딕셔너리로 만들기

        print(j, tag_list)

        # 다음장 클릭
        CSS_tran2="body > div.RnEpo._Yhr4 > div.Z2Inc._7c9RR > div > div.l8mY4.feth3 > button > div > span > svg"             # 다음 버튼 정의
        tran_button2 = driver.find_element_by_css_selector(CSS_tran2)  # 다음 버튼 find
        AC(driver).move_to_element(tran_button2).click().perform()     # 다음 버튼 클릭
        time.sleep(3)

    except:  # 에러가 나면 다음장을 클릭해라
        # 다음장 클릭
        CSS_tran2="body > div.RnEpo._Yhr4 > div.Z2Inc._7c9RR > div > div.l8mY4.feth3 > button > div > span > svg"             # 다음 버튼 정의
        tran_button2 = driver.find_element_by_css_selector(CSS_tran2)  # 다음 버튼 find
        AC(driver).move_to_element(tran_button2).click().perform()     # 다음 버튼 클릭

        time.sleep(3)

print(dict)
# 판다스 데이터프레임으로 만들기 : 엑셀(테이블) 형식으로 만들기
import pandas as pd
result_df = pd.DataFrame.from_dict(dict, 'index')

print(result_df.shape)
result_df
result_df.to_csv("insta({}).csv".format(keyword), encoding='utf-8-sig')
```
```py
# 이미지들 image_insta 폴더에 다운받기
import os
import urllib.request

# 만약 image_insta 폴더가 없으면 만들어라
if not os.path.exists("image_insta"):
    os.makedirs("image_insta")
        
for i in range(0, 50):
    
    try:
        index = result_df['picture'][i]
        date = result_df['date'][i]
        urllib.request.urlretrieve(index, "image_insta/{0}_{1}.jpg".format(date, i))        
    except:
        pass

```
- 사진 다운받음