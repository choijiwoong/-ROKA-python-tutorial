    #유클리드 거리(Ecuclidean distance)는 다차원 공간(DTM)에서 두 점 사이의 거리를 계산하는 것이다.
import numpy as np

def dist(x, y):
    return np.sqrt(np.sum((x-y)**2))

doc1=np.array((2,3,0,1))#4차원 공간에서의 유클리드 거리
doc2=np.array((1,2,3,1))
doc3=np.array((2,1,2,2))
docQ=np.array((1,1,0,1))#target

print('문서1과 문서Q의 거리: ', dist(doc1, docQ))#more similar!
print('문서2와 문서Q의 거리: ', dist(doc2, docQ))
print('문서3과 문서Q의 거리: ', dist(doc3, docQ), end='\n\n')


    #자카드 유사도(Jaccard Similarity)는 합집합에서의 교집합 비율을 구하는 것이다. 집합이 동일하면 1, 교집합이 없다면 0의 값을 갖는다.
doc1="apple banana everyone like likey watch card holder"
doc2="apple banana coupon passport love you"

tokenized_doc1=doc1.split()
tokenized_doc2=doc2.split()

print('문서1: ', tokenized_doc1)
print('문서2: ', tokenized_doc2)

#get 합집합
union=set(tokenized_doc1).union(set(tokenized_doc2))#set연산을 이용
print("문서1과 문서2의 합집합(union): ", union)

#get 교집합
intersection=set(tokenized_doc1).intersection(set(tokenized_doc2))
print('문서1과 문서2의 교집합: ', intersection)

#calculate Jaccard Similarity
print('자카드 유사도: ', len(intersection)/len(union), end='\n\n')


#(p.s)레벤슈타인 거리 알고리즘(Levenshtein Diatance)알고리즘은 두 문자열이 같아지려면 몇번의 문자 조작(삽입, 삭제, 변경)이 필요한지 구하는 것이다.
#편집거리를 구하여 기존의 단어 리스트와 가장 작은 편집거리를 가진 단어를 추천하는 서비스를 제공해줄 수 있다. https://peanut159357.tistory.com/77
import numpy as np
def levenshtein(seq1, seq2):
    size_x=len(seq1)+1
    size_y=len(seq2)+1
    matrix=np.zeros((size_x, size_y))#0으로 초기화

    for x in range(size_x):#기본적으로 초기화 할 수 있는 것들은 초기화
        matrix[x,0]=x
    for y in range(size_y):
        matrix[0,y]=y

    for x in range(1, size_x):#위에 초기화된거 제외
        for y in range(1, size_y):
            if seq1[x-1]==seq2[y-1]:#두 seq에서의 현재 편집거리가 같다면
                matrix[x,y]=matrix[x-1,y=1]#문자삽입, 제거, 변경 중에 최소편집거리로 matrix채워줌
            else:
                matrix[x,y]=min(matrix[x-1,y]+1, matrix[x-1,y-1]+1, matrix[x,y-1]+1)#대각선, 좌측, 위중 가장 작은값에 +1한 값
    return matrix[size_x-1, size_y-1]#제일 오른쪽 아래값 반환. 서로 같아지기 위한 연산의 횟수.
seq1="THESTRINGS"
seq2="THISISIT"
print("seq1과 swq2의 levenshtein거리: ", levenshtein(seq1,seq2))#6개의 글자를 고쳐야 한다.
