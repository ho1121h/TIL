# DJANGO 게시판 만들기 실습
1. 블로그 생성
```PY
python manage.py startapp blog
```
로 블로그 생성(APP)
2. setting.py 에 INSTALLED_APPS에 등록

3. .blog에서 view 생성 (컨트룰)
- 게시글을 조회 하는 함수생성
```py
#블로그 컨트롤
def blog_list(request):
    # Django-ORM은 DB를 조회할때 매니저 객체를 사용
    # Model 에서 Manager객체에 접근 -> 모델 오브젝트 / <Model>.objects
    #https://docs.djangoproject.com/en/4.0/ref/models/querysets/

    #1. 전체 게시글 조회하기 (SELECT * FROM blog_post) 터미널창에 게시글이 조회됨
    #post_list = Post.objects.all()
    #print(post_list)

    # 2. 필터걸기( where 절 사용)
    #post_list = Post.objects.filter(id=1).all()
    #print(post_list)

    # 3. insert(데이터 추가하기)
    #Post.objects.create(
    #    title = "게시글4",
    #    content = "4번째 게시글 입니다"
    #)

    # 전체 게시글을 조회해서 , blog_list.html에 리스트를 rendering 하세요
    #(심화) pagination을 구현하세요
    post_list = Post.objects.all()

    return render(request, 'blog_list.html',{'post_list':post_list})

```
- 내용을 자세히 조회
```py
#url : /list/<blog_id>  고로 파라미터에 추가
def blog_detail(request, blog_id):
    #블로그 내용을 조회하고 보여주는 페이지 작성(blog_detail.html)
    #(심화) : blog_list 에서 리스트르 클릭하면 해당 페이지로 이동할 수 있도록 하셈

    
    blog = Post.objects.get(id=blog_id)
    
    return render(request,'blog_detail.html',{'blog':blog})

```
- 게시글을 작성하는 함수 생성
```py
def blog_post(request):
    form =PostForm()
    if request.method =='POST':
        form = PostForm(request.POST)
        if form.is_valid(): 
            form.save()
        # data = request.POST
        # board = Post.objects.create(title = data['title'],content=data['content'])
            return redirect("blog_list")

    return render(request, "blog_post.html",{'form': form})
```
- form 을 쓰면 게시글 코드짜기 편해진다
- .blog 에 forms.py 를 만들고 
- pip install django-crispy-forms
- pip install crispy-bootstrap5
- 를 설치하자

```py
#forms.py
from attr import field
from django import forms
from .models import Post

#  좀더 세련된 방법

# Form 을 파이썬에서 관리함
# 대체로 모델과 비슷하다
# 입력을 받는 것은 DB에서 넣고 받기 때문

# class PostForm(forms.Form):
#     title = forms.CharField(max_length=255)
#     content = forms.CharField(widget=forms.Textarea)

#https://docs.djangoproject.com/en/4.0/topics/forms/modelforms/

class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['title','content']


```
4. urls.py 에 경로를 설정하자 

```py
urlpatterns = [
    path('', views.index, name='index'),
    path('list/', views.blog_list, name='blog_list'),#url , 함수이름(뷰에 있음), 이름
    path('list/<int:blog_id>', views.blog_detail, name='blog_detail'),
    path('write', views.blog_post, name = 'blog_post')
]

```

5. html을 만들자 
- 관리를 쉽게하기 위해 base 생성 ,홈화면,  블로그 리스트, 블로그 생성, 블로그상세를 만든다.
---
베이스
```html
<!-- base.html -->
    <main class="flex-shrink-0"> 
        {% block main %}
        
        {% endblock %}
    </main>
<!-- base.html -->
```
---
홈화면
```html
{% extends "base.html" %}
{% block main %}
    <div class="container">

        <!--넣고 싶은 템플릿 -->
    </div>
{% endblock %}
```
---
블로그 리스트(게시글 리스트)
```html
{% extends "base.html" %}
{% block main %}
<div class="container">
    <h1 class="mt-5">블로그</h1>       
    <div class="list-group">
      <style>
        .list-group-item:hover {
            z-index: 2;
            color: var(--bs-list-group-active-color);
            background-color: var(--bs-list-group-active-bg);
            border-color: var(--bs-list-group-active-border-color);
        }
      </style>
      <div><!--{% url 'blog_post' %} 은 해당 주소에서 파라미터를 가져옴 -->
        <a href="{% url 'blog_post' %}" class="btn btn-primary float-end"> 글 쓰기 </a>
      </div>
    <!--불러온 파라미터를 for문으로 반복해서 게시글 전체 출력-->
      {% for post in post_list %}
      <a href="{% url 'blog_detail' post.id %}" class="list-group-item list-group-item-action " aria-current="true">
        <div class="d-flex w-100 justify-content-between">
          <h5 class="mb-1">{{ post.title }}</h5>
          <small>{{ post.created_at }}</small>
        </div>
        <p class="mb-1">{{ post.content }}</p>
        <!--<small>And some small print.</small>-->
      </a>
      {% endfor %}
    </div>
</div>
{% endblock %}

{% block extra_css %}
       <!--style>
            .footer{
                background-color: violet !important;
            }

        </style>--> 
{% endblock %}


```
---
- 게시글 상세

```html
{% extends "base.html" %}
{% block main %}
    <div class="container">
        <h1 class='my-5'>{{ blog.title }}</h1>
        <div class="p-5 border-2" style="border: 1px solid #cdcdcd;">
            {{ blog.content }}
        </div>
    </div>
{% endblock %}

```
---
- 게시글 작성 form 모듈이 알아서 해주도록 설정

```html
{% extends "base.html" %}
{% load crispy_forms_tags %}

{% block main %}
  <div class="container">
    <form method="POST">
      {% csrf_token %}
      
      {{ form | crispy }}
      <button type = 'submit'> 제출 </button>
    </form>
  </div>
  
{% endblock %}



```
## 실행 
```py
python manage.py runserver

```