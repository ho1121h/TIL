#10988 펠린드롬 문제
word=str(input())
if word[:] == word[::-1]:
    print(1)
else:
    print(0)


#2711
 
n=int(input())#테스트 케이스의 갯수
for i in range(n):
    t,s=input().split()
    print(s[:int(t)-1]+s[int(t):])

#17249

s,s1=map(str,input().split('(^0^)'))
print(list(s).count("@"),list(s1).count("@"))

#2789
word=input()
del_word="CAMBRIDGE"
for i in range(len(del_word)):
    word=word.replace(del_word[i],"")
print(word)
