#[2439](https://www.acmicpc.net/problem/2439)
N=int(input())
for i in range(1,N+1):
    print(' '*(N-i),'*'*i,sep="")
