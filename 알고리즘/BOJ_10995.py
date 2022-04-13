N=int(input())
for i in range(N):
    if i%2 ==1:#2로 나눠 1이 몫으로 나올 경우 다음을 실행
        print("",end=" ")#첫문자를 공백으로 출력 하고 for문을 실행
    for j in range(N):#매개변수 만큼 *을 출력하나 끝에 띄움
        print("*",end=" ")
    print()