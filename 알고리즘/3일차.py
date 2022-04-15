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
    
    for word, digit in numbers.items():
        s = s.replace(word, digit)
    
    return int(s)