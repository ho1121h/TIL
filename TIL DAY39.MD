# Django 실습
--- 
- 한국은 자바공화국이다. 하지만 나는 지금 배우고 있는 과정에서는 파이썬만을 배웠다. 언젠간 웹개발도 할지 모르니, 파이썬으로 웹개발을 할 수 있는 장고를 배워보기로 한다... (독학으로)

---
## 설치방법
설치 방법은 간단하다 터미널로 설치해주자
```py
pip install django # 인터프리터 파이썬 사용시
```
- 이제 장고서버 파일을 만들자
- 만들고자 하는 폴더에 git bash를 이용하여 폴더를 생성한다. 그러면 장고에 관한 파일들이 자동 생성된다.<프로젝트_파일>
```
$ django-admin startproject mysite
```
- 생성후, 터미널창에 입력하면 서버가 열린다.
```
python manage.py runserver

```
## 설치 후 어플리케이션 생성

일단 모델-뷰-컨트롤러 (MVC)의 개념을 이해해야한다.
- 모델: 서버의 데이터 베이스
- 뷰: 브라우저 상 사용자에게 보여주는 화면
- 컨트롤러: 모델에 일을 시키는 명령어,작업요청

---
쟝고에서는 어플리케이션 단위로 일을 시키기 때문에 어플리케이션을 생성하기로한다.
커맨드창에 다음을 입력해서 폴더를 생성한다.
```
python manage.py startapp polls
```
그러면 polls 라는 폴더가 생기고 
폴더 안의 veiw.py에 들어가서 다음과 같이 입력한다. 해당서버에 들어오면 다음과 같은 문구가 출력될 것이다.
```py
from django.shortcuts import render
from django.http import HttpResponse


def index(request):
    return HttpResponse("Hello, World. You're at the polls index.")
```
출력을 하기 위해서 이 코드를 url에 등록을 해야한다.
view코드가 있는자리에 urls.py 를 생성한다

```py
from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
]
```
그러고 난 뒤 main에 연결하면 이작업은 마무리가된다.
우선 mysite 폴더 안에 urls.py에 들어가고 

```py
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('polls/', include('polls.urls')),
    path('admin/', admin.site.urls),
]

```
로 수정 한뒤 http://127.0.0.1:8000/polls/ 로 접속하면 어플리케이션대로 출력이 된다.
- polls에 include를 연결한단 뜻

## 모델 작성 

migrations 의 모델에 모델코드를 작성을하자.

- 예시로 투표시스템을 작성해야하므로 다음과 같이 작성
```py
from django.db import models

# Create your models here.
# 서로 외래키라서 한쪽이 지워지면 같이 지워짐

class Question(models.Model): #질문
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')

class Choice(models.Model): #선택
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200) #문자열 필드와 글자수 제한
    votes = models.IntegerField(default=0) # 정수형 필드와 기본값 0
```
## 모델 활성화
앞서 프로젝트에 어플리케이션을 연결을 해주었다.-> 모델생성
그런데, 어플리케이션의 모델도 프로젝트 모델에 연결해야한다.
메인 폴더의 셋팅 코드에 INSTALLED_APPS 칸에 'polls.apps.PollsConfig'를 등록해 주자.
```py 
from django.apps import AppConfig


class PollsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'polls'

```
를 연결해주는것이다.
터미널창에 다음을 입력하자.
>>python manage.py makemigrations polls
그러면 migrations폴더가 생성될 것이다.

그다음 터미널창에 
>>python manage.py migrate 
를 입력하면 우리들의 메인 프로젝트에 모델 정보가 등록이된다.
즉, 모델의 정보가 갱신됨
- 모델의 정보를 보고싶다면 "python manage.py sqlmigrate polls + 변경기록번호"를 입력