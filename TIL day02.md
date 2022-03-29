## gitignoer
깃 이그노어를 만들어보자
- commit 을 하기전에 README.md 와 .gitignore 파일을 만든다
- .gitignore 파일안에 깃 허브에 업로드 하지않을 파일명을 적는다.


## git clone
1. git clone 은 깃허브에 저장된 파일을 다운 할 수 있다.
2. 다운받을 위치에 git bash를 킨다.
3. CLI에 git clone URL 을 입력한다.
- git clone을 통해 생성된 로컬 저장소는 git init과 git remote add가 이미 수행되어 있다. 

## pull
- pull 명령어는 협업할때 많이 쓰이게 된다. 새로운 버전으로 업데이트하는 개념
- 명령어: git pull 별명 master
- 풀(당기고)로 업데이트하고 푸쉬(밀어넣기)로 업로드를 다시하면된다.

## branch
- 브랜치는 독립 공간을 형성한다. 원본으로 부터 독립되었기 때문에 수정으로 원본 훼손에 대해 안전하다.
- `git branch` 브랜치 목록 확인
- `git branch 브랜치이름` 새로운 브랜치 생성
- `git branch -d 브랜치이름` 브랜치 삭제 -D 는 강제 삭제
- `git switch 다른브랜치 이름` 다른 브랜치로 HEAD를 이동
- `git switch -c 브랜치이름` 생성과 동시에 이동
- 주의점은 브랜치를 이동하기전에 커밋을 완료하고 이동해야 한다.

## merge
- branch 로 여러 버전을 만들고 합치는 과정
- `git branch merge 현재 해드`
- `git graph` 로 위치 확인 