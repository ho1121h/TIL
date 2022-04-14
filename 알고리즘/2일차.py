#11720 숫자의 합
n = int(input())#그냥 입력 의미없는 숫자
print(sum(map(int,input())))#두번째 입력할 값(나열)을 각각 정수원소로 취급하여 더함
#map(적용시킬 함수, 함수를 적용할 값들)

#2750 수 정렬하기 (내장함수 활용)
n = int(input())#범위 지정할 정수 입력
n_1 = []#리스트 추가
for i in range(n):#0부터 반복할 범위 설정
    n_1.append(int(input()))#범위 만큼 원소 갯수 각각 입력
n_2 = sorted(n_1)#리스트를 새로 정렬
for i in range(len(n_1)):#리스트 크기만큼 반복
    print(n_2[i])#리스트를 인덱싱하여 하나 씩 추가

# List Comprehension http://jungol.co.kr/bbs/board.php?bo_table=pbank&wr_id=4348&sca=pyc0
number=[i**2 for i in range(1,int(input())+1)]
print(number)

# List Comprehension http://jungol.co.kr/bbs/board.php?bo_table=pbank&wr_id=4356&sca=pyc0
n = int(input())
numbers = [f"No.{i}" for i in range(1, n + 1)]
print(numbers)

# 9줄 입력 ,최댓값, 최댓값 자릿수
number = [int(input()) for i in range(9) ]
print(max(number))
print(number.index(max(number))+1)

# http://jungol.co.kr/bbs/board.php?bo_table=pbank&wr_id=4353&sca=pyc0
#frist 입력
line=[list(map(int,input().split())) for i in range(2)]

#second 입력
line2=[list(map(int,input().split())) for j in range(2)]

for i in range(2):
    for j in range(3):
        print(line[i][j] * line2[i][j], end=" ")
    print()

'''
3 6 9
8 5 2
9 8 7
6 5 4
'''
list_a = [list(map(int, input().split())) for i in range(2)]
list_b = [list(map(int, input().split())) for i in range(2)]
list_c = [[0] * 3 for _ in range(2)]
for i in range(2):
    for j in range(3):
        list_c[i][j] = list_a[i][j] * list_b[i][j]
for i in range(2):
    for j in range(3):
        print(list_c[i][j], end=" ")
    print()
