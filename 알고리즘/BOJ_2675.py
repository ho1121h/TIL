t=int(input())
for i in range(t):
    R,S=map(str,input().split())
    S=list(S)
    for j in range(len(S)):
        print(S[j]*int(R),end="")
    print()