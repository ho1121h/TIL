#[11022](https://www.acmicpc.net/problem/11022)
T=int(input())
for i in range(T):
    A,B=map(int,input().split())
    print(f"Case #{i+1}: {A} + {B} = {A+B}")