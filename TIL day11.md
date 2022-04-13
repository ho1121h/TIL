## 알고리즘 특강
1. [1000번](https://www.acmicpc.net/problem/1000)
```python
A,B=map(int,input().split())
print(A+B)
```
2. [2558번](https://www.acmicpc.net/problem/2558)
```python
A=int(input())
B=int(input())
print(A+B))
```
3. [10950](https://www.acmicpc.net/problem/10950)
```python
T=int(input())
for i in range(T):
    A,B=map(int,input().split())
    print(A+B)
```
4. [10953](https://www.acmicpc.net/problem/10953)
```python
T=int(input())
for i in range(T):
    A,B=map(int,input().split(","))
    print(A+B)
```
5. [11021](https://www.acmicpc.net/problem/11021)
```python
T=int(input())
for i in range(T):
    A,B=map(int,input().split())
    print(f"Case #{i+1}: {A+B}")
```
6. [11022](https://www.acmicpc.net/problem/11022)
```python
T=int(input())
for i in range(T):
    A,B=map(int,input().split())
    print(f"Case #{i+1}: {A} + {B} = {A+B}")
```
7. [2438](https://www.acmicpc.net/problem/2438)
```python
N=int(input())
for i in range(1,N+1):
    print("*"*i)
```
8. [2439](https://www.acmicpc.net/problem/2439)
```python
N=int(input())
for i in range(1,N+1):
    print(' '*(N-i),'*'*i,sep="")
```
