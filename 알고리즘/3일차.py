#카카오 기출 문제 https://programmers.co.kr/learn/courses/30/lessons/81301?language=python3
#딕셔너리 활용가능
#enumerate 활용가능
def solution(s):
    numbers = {
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9'
    }
    #   키    밸류   는 딕셔너리에서 items매소드로 가져옴
    for word, digit in numbers.items():
        s = s.replace(word, digit)
    
    return int(s)

## 파울 선수 목록 입력 후 가장 적게 친 선수 출력 및 파울 갯수출력
players = input().split()
fouls = {}

for player in players:
    # 1. 파울 목록에 이름이 이미 있다
    if player in fouls:
        fouls[player] += 1
    # 2. 파울 목록에 이름이 없다
    else:
        fouls[player] = 1

min_foul = min(fouls.values()) # 2

for player, foul in fouls.items():
    if foul == min_foul:
        print(player)

print(min_foul)

## 딕셔너리에서 밸류값 찾기
n=int(input())
Country = {}

for _ in range(n):
    m, n=map(str,input().split())
    Country[m] = f"{n}"

find_Country=str(input())
print(Country.get(find_Country,"Unknown_Country"))