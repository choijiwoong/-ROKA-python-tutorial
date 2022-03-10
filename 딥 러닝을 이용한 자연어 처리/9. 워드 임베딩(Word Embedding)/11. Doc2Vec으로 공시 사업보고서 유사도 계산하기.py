#Doc2Vec은 Word2Vec을 변형하여 문서의 임베딩을 얻을 수 있도록 한 알고리즘으로 Gensim을 통해 사용이 가능하다.
    #[1. 공시 사업 보고서 로드 및 전처리]
import pandas as pd
from konlpy.tag import Mecab
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm

df=pd.read_csv('dart.csv', sep=',')#https://doc-04-7k-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/sfqek6g16gim188f1nsfb1dmvsuiekrm/1646918625000/17609157229046208934/*/1nIUZxASrbIa0Z2xzGVgX1dqjtnFCoKNT?e=download
df=df.dropna()#결측값 미리 제거
print("Dart data:\n", df)#bussiness열에 사업보고서 저장되어 있음

mecab=Mecab('C:/Users/admin0!/AppData/Local/Programs/Python/Python39/Lib/site-packages')#형태소 분석기
tagged_corpus_list=[]
for index, row in tqdm(df.iterrows(), total=len(df)):#이 for문이 뭘 의미하는지 모르겠음..놀랍게도 tqdm의 인자라니...반복의 총 획수를 나타낸다함.
    text=row['business']#필요한 data추출 
    tag=row['name']
    tagged_corpus_list.append(TaggedDocument(tags=[tag], words=mecab.morphs(text)))#리스트에 TaggedDocument를 append. 이는 데이터에 태그를 달아 접근가능하게 함. 제목과 tokenized된 형태소 토큰을 저장.
print('문서의 수: ', len(tagged_corpus_list))

print('(test)첫번째 문서의 전처리 결과: ', tagged_corpus_list[0])

    #[2. Doc2Vec 학습 및 테스트]
from gensim.models import doc2vec

model=doc2vec.Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.025, workers=8, window=8)#Doc2Vec의 모델 인스턴스화. (alpha는 초기 학습률, min_alpha는 학습이 진행되며 떨어뜨릴 학습률을 의미한다.)

model.build_vocab(tagged_corpus_list)#TaggedDocument로 vocab빌드
print('Tag Size: ', len(model.docvecs.doctags.keys()), end=' / ')

model.train(tagged_corpus_list, total_examples=model.corpus_count, epochs=50)

model.save('dart.doc2vec')

#test
similar_doc=model.docvecs.most_similar('동화약품')
print("동화약품과 사업보고서가 유사한 회사들: ", similar_doc)

similar_doc=model.docvecs.most_similar('삼성전자')
print("삼성전자와 사업보고서가 유사한 회사들: ", similar_doc)
