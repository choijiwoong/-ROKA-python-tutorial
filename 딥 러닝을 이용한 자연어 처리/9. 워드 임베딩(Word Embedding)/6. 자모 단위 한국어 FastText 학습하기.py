#자소분리된 코퍼스 준비->테스트 문자를 자소로 분리하여 전달->결과자소를 합쳐 원래 단어로 변경의 과정을 따른다. https://joyhong.tistory.com/137
 #utils
import re
from soynlp.hangle import compose, decompose, character_is_korean#초중종결합, 초중종분리, 한국어가 맞는지(초중종각각 범위내에서 ck)

doublespace_pattern=re.compile('\s+')

def jamo_sentence(sent):
    def transform(char):#' '->' ', decompose결과가 1개면 반환, decompose결과가 공백이면 -로 바꿔 join
        if char==' ':
            return char
        cjj=decompose(char)
        if len(cjj)==1:
            return cjj
        cjj_=''.join(c if c!=' ' else '-' for c in cjj)
        return cjj_

    sent_=[]
    for char in sent:
        if character_is_korean(char):#한국어는 초중성분리하여 append
            sent_.append(transform(char))
        else:
            sent_.append(char)
    sent_=doublespace_pattern.sub(' ',''.join(sent_))#doublespace는 하나로
    return sent_

def jamo_to_word(jamo):
    jamo_list, idx=[], 0
    while idx<len(jamo):
        if not character_is_korean(jamo[idx]):#한국어아니면 append
            jamo_list.append(jamo[idx])
            idx+=1
        else:
            jamo_list.append(jamo[idx:idx+3])#한국어면 3개(초중종)한번에 append
            idx+=3

    word=""
    for jamo_char in jamo_list:
        if len(jamo_char)==1:#일반 char(non-Korean)
            word+=jamo_char
        elif jamo_char[2]=='-':#종성x
            word+=compose(jamo_char[0], jamo_char[1], ' ')#compose를 위해 다시 공백으로
        else:
            word+=compose(jamo_char[0], jamo_char[1], jamo_char[2])
    return word#compose된 word반환
            

 #corpus_mecab.txt데이터 자소분리
from tqdm import tqdm

def process_jamo(tokenized_corpus_fname, output_fname):#자모 tokenize
    total_lines=sum(1 for line in open(tokenized_corpus_fname, 'r' ,encoding='utf-8'))#total_line체크

    with open(tokenized_corpus_fname, 'r', encoding='utf-8') as f1, open(output_fname, 'w', encoding='utf-8') as f2:
        for _, line in tqdm(enumerate(f1), total=total_lines):
            sentence=line.replace('\n', '').strip()#하나의 라인에서 \n제거
            processed_sentence=jamo_sentence(sentence)#jamo_decompose화
            f2.writelines(processed_sentence+'\n')#파일에 쓰기.
tokenized_corpus_fname='C:/Users/admin0!/Desktop/_2jimo/python/딥 러닝을 이용한 자연어 처리/9. 워드 임베딩(Word Embedding)/tokenized/corpus_mecab_jamo.txt'
output_fname='C:/Users/admin0!/Desktop/_2jimo/python/딥 러닝을 이용한 자연어 처리/9. 워드 임베딩(Word Embedding)/tokenized/corpus_mecab_jamo.txt'
process_jamo(tokenized_corpus_fname, output_fname)

 #fasttext skip-gram model 학습
from gensim.models import FastText
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)#그냥 로깅해주는거같고.

print('corpus 생성')
corpus=[sent.strip().split(' ') for sent in tqdm(open(tokenized_corpus_fname, 'r', encoding='utf-8').readlines())]#' '단위 읽어온 데이터 리스트 저장

print('학습 중')
model=FastText(corpus, vector_size=100, workers=4, sg=1, iter=2, word_ngrams=5)
model.save(model_fname)

print('학습 소요 시간:',model.total_train_time)
model.wv.save_word2vec_format(model_fname+'_vis')#projector.tensorflow.org 시각화를 위한 별도 저장..?뭐 그런게 있나보네 pyplot시각화 사이트처럼
print('완료')

 #이용
from gensim.models import FastText

def transform(list):
    return [(jamo_to_word(w), r) for (w, r) in list]

loaded_model=FastText.load("C:/Users/admin0!/Desktop/_2jimo/python/딥 러닝을 이용한 자연어 처리/9. 워드 임베딩(Word Embedding)/tokenized/ratings_soynlp.txt")
print("loaded_model의 shape: ", loaded_model.sv.vectors.shape)

print(transform(loaded_model.wv.most_similar(jamo_sentence('최민식'), topn=5)))
