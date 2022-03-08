    #1. 위키피디아로부터 데이터 다운로드 및 통합
import os
import re

#위키피디아 한국어 덤프는 총 6개의 디렉토리(AG, AI, ...)로 90여개의 파일(wiki_00~wiki_)이 있다.
print('text디렉토리 안의 디렉토리 확인: ', os.listdir('text'))

def list_wiki(dirname):#AA~AF(6개)의 디렉토리안의 모든 파일들의 경로(address)를 파이썬의 리스트 형태로 저장.
    filepaths=[]
    filenames=os.listdir(dirname)#dir리스트를 가져와서
    for filename in filenames:#그 filename만큼
        filepath=os.path.join(dirname, filename)#인자로 들어온 dirname+filename문자열을 concat하고,

        if os.path.isdir(filepath):#그 주소가 directory라면
            filepaths.extend(list_wiki(filepath))#해당 디렉토리를 재귀로 directory가 아닐때까지 돌리다가 filepaths에 추가
        else:
            find=re.findall(r"wiki_[0-9][0-9]", filepath)#regex 형식을 따르는 filename을 모두 찾아
            if 0<len(find):
                filepaths.append(filepath)#filepaths에 append.
    return sorted(filepaths)#정렬된 파일경로를 리턴.
filepaths=list_wiki('text')
print("text파일의 개수: ", len(filepaths))#850

with open("output_file.txt", 'w') as outfile:#filepaths의 파일내용을 output_file.txt에 통합하여 저장.
    for filename in filepaths:
        with open(filename) as infile:
            contents=infile.read()
            outfile.write(contensts)

f=open('output_file.txt', encoding='utf8')

i=0
while True:#output_file.txt의 10줄만 출력(테스트..)
    line=f.readline()
    if line != '\n':
        i=i+1
        print('%d번째 줄: '%i+line)
    if i==10:
        break
f.close()

    #2. 형태소 분석
from tqdm import tqdm
from konlpy.tag import Mecab

mecab=Mecab()

f=open('output_file.txt', encoding='utf8')
lines=f.read().splitlines()
print("output_file의 line수: ", len(lines))
print('lines의 상위 10개 출력: ', lines[:10])

result=[]
for line in tqdm(lines):
    if line:
        result.append(mecab.morphs(line))#형태소 분석하여 공백line제거하며 result에 append
print("형태소 분석 결과의 길이(문장 수): ", len(result))

    #3. Word2Vec 학습
from gensim.models import word2vec
model=word2vec(result, vector_size=100, window=5, mim_count=5, workers=4, sg=0)

model_result1=model.wv.most_similar("대한 민국")
print("대한민국과 유사한 단어: ", model_result1)
