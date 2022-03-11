    #[1. 네이버 영화 리뷰 데이터에 대한 이해와 전처리]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

 #1. 데이터 로드하기
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
train_data=pd.read_table('ratings_train.txt')
test_data=pd.read_table('ratings_test.txt')

print('훈련용 리뷰 개수: ', len(train_data))
print('(test)상위 5개 훈련용 리뷰 출력:', train_data[:5])#id는 sentiment classification에 도움이 안되는 데이터임을 알 수 있다!
print('(test)상위 5개 테스트용 리뷰 출력:', test_data[:5],'\n\n')#또한 한국어 특성상 띄어쓰기도 잘 안돼있고 오타도 많고 신조어, 줄임말도 많음을 알 수 있다. 좆됬다 근데 천재가 이미 만들어놓은거 전에 배웠다ㅋ

 #2. 데이터 정제하기
print('훈련데이터의 중복유무 확인: ', train_data['document'].nunique(), ', 훈련데이터 label 중복유무 확인??ㅋㅋ왜하노: ', train_data['label'].nunique())
#무튼 총 150000샘플이 train_data로 존재하는데, 중복제거시 146182개니까 총 4000개정도의 중복샘플이 존재한다는 것이기에 삭제한다.
train_data.drop_duplicates(subset=['document'], inplace=True)#document열 기준 중복되는 데이터 제거
print('\ndocument열 기준 중복 제거 후 총 샘플의 수:', len(train_data),'\n')

#데이터 분포 확인
train_data['label'].value_counts().plot(kind='bar')
plt.show()#균일해보인다.
print(train_data.groupby('label').size().reset_index(name='count'),'\n')#label0이 근소하게 많다

print('리뷰중에 Null값 유무:', train_data.isnull().values.any(),'\n')
print('Null이 어떤 열에 존재하는지 확인: \n', train_data.isnull().sum(),'\n')#만약 필요없는 id열이 null이면 굳이 처리할 필요가 없기에. 하지만 document에 1개 확인
print("document열에 Null값을 가진 샘플이 어느 index에 위치해있는지: \n", train_data.loc[train_data.document.isnull()],'\n')#document null인 항목(document)의 location.

train_data=train_data.dropna(how='any')#위에 어느 인덱스고 어느 열이고는 그냥 공부 차원에서 해본거고 null있는거 다 삭제
print('dropna실행 후 null값이 존재하는지 확인: ', train_data.isnull().values.any())
print('null값 제거 후 총 샘플의 개수: ', len(train_data),'\n')

#특수문자를 없애는데에 영어 re.sub(r'[^a-zA-Z ]', '', eng_text)와 유사하게 처리가 가능하다.
train_data['document']=train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
print("(test)regex이용 한글만 남긴 상위 5개 리뷰:\n", train_data[:5],'\n')

#만약 영어로만 이루어진 리뷰일 경우 한국어만 필터링하면서 NaN값이 되었을 가능성이 농후하니 null값을 다시 확인한다.
train_data['document']=train_data['document'].str.replace('^ +', "")#double space이상을 empty value로 변경
train_data['document'].replace('', np.nan, inplace=True)#empty value를 NaN으로 변경(두번에 걸친 이유는 일반 space마저도 empty value로 바꾸면 정상문장 사이 띄어쓰기가 사라지기에)
print('한글 필터링 이후 null값의 유무 확인:')
print(train_data.isnull().sum(),'\n')#789개 null데이터 형성.

print("(test)Null이 있는 상위 5개 행 출력: ")
print(train_data.loc[train_data.document.isnull()][:5])
#제거
train_data=train_data.dropna(how='any')
print("한글 필터링 후 null값 제거한 뒤의 총 샘플의 개수: ", len(train_data),'\n\n')#145393

 #3. 토큰화_java.lang.java.lang.ExceptionInInitializerError: java.lang.ExceptionInInitializerError으로 지금부터 Colab사용
stopwords=['의 ', '가', '이', '은', '들', '는', '좀', '잘',' 걍', '과', '도', '를', '으로', '자', '에 ','와 ','한 ','하다']#대충 이정도만 사용
okt=Okt()
print("(test)okt 형태소분석기 테스트:", okt.morphs('와 이런 것도 영화라고 차라리 뮤직비디오를 만드는 게 나을 뻔', stem=True))

#토근화, 불용어제거
X_train=[]
for sentence in tqdm(train_data['document']):
    tokenized_sentence=okt.morphs(sentence, stem=True)
    stopwords_removed_sentence=[word for word in tokenized_sentence if not word in stopwords]
    X_train.append(stopwords_removed_sentence)
print('\n(test)토큰화, 불용어제거된 상위3개의 샘플: ')
print(X_train[:3])

X_test=[]#테스트데이터도 같이적용
for sentence in tqdm(test_data['document']):
    tokenized_sentence=okt.morphs(sentence, stem=True)
    stopwords_removed_sentence=[word for word in tokenized_sentence if not word in stopwords]
    X_test.append(stopwords_removed_sentence)

 #4. 정수 인코
딩
