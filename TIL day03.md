## Undoing
- 파일 내용을 수정전으로 돌리는 법 예시
1. a.txt 생성
2. git add .
3. git commit -m "first commit"
4. git log --oneline 
> 여기 까진 기본적으로 생성 해보는것
5. git restore a.txt
> a.txt 의 내용을 수정 후 저장하고 저장 전 상태로 돌리는 명령어
6. git restore --staged a.txt (커밋 O) 
> 스테이지에 깃의 위치를 작업 영역으로 끌어내리는것 단, 커밋을 했을 경우의 상황. 1~3번 을 했을 때
7. git rm --cached a.txt (커밋 X)
> 스테이지에 깃의 위치를 작업 영역으로 끌어내림 단, 커밋을 하기 전
8. git commit --amend
> 깃을 커밋하고 커밋 메세지를 편집하는 명령어
-> vim 화면 나타나면 i 를 누른다
-> 그 상태에서 편집한다
-> 편집 후 esc를 누르고 :wq 누른다
-> 엔터를 누르면 완료!

---
## 버전을 되돌리기
- git reset --soft {해시값} 
> 깃의 버전을 커밋하기전(스테이지 상태)로 되돌린다.
- git reset --mixed {해시값}
> 깃의 버전을 add 하기전(작업 영역)으로 되돌린다.
- git reset --hard {해시값}
> 해당 버전을 삭제한다.
---
## git reflog
- git reflog
> 삭제포함해서 전 기록을 보여준다.