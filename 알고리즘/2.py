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