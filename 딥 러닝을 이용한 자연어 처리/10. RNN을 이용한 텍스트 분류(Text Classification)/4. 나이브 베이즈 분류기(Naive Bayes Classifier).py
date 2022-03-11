""" 텍스트 분류에 사용되는 전통적인 분류기로, 인공 신경망 알고리즘은 아니지만 준수한 성능을 보인다.
    1. 베이즈의 정리(Bayes's theorem)을 이용한 분류 메커니즘
조건부 확률을 계산하는 방법 중 하나로, P(A|B)=P(B|A)P(A)/P(B)로 쉽게 P(B|A)를 구하는 방법이다. 이를 메일분류에 적용하면 아래와 같다.

P(정상메일|입력텍스트)=입력텍스트가 있을때 정상메일일 확률
P(스팸메일|입력텍스트)=입력텍스트가 있을때 스팸메일일 확률
이에 베이즈정리를 적용하면
P(정상메일|입력텍스트)=(P(입력텍스트|정상메일)xP(정상메일))/P(입력텍스트)
P(스팸메일|입력텍스트)=(P(입력텍스트|스팸메일)xP(스팸메일))/P(입력텍스트)
를 간소화하여 /P(입력텍스트)를 간소화 할 수 있고, 결정은 P(정상메일|입력텍스트)와 P(스팸메일|입력텍스트)의 비교로 결정할 수 있다.
이를 단어(w1, w2, w3) 토큰화하여 입력으로 사용한다면
P(정상메일|입력텍스트)=P(w1|정상메일)xP(w2|정상메일)xP(w3|정상메일)xP(정상메일) (쉽게 말해 정상메일x정상메일에w1x정상메일에w2해서 단어하나하나의 확률을 곱하여 문서의 확률을 구함)
P(스팸메일|입력텍스트)=P(w1|스팸메일)xP(w2|스팸메일)xP(w3|스팸메일)xP(스팸메일)
즉, BoW와 같이 순서 없이 빈도수만 고려한다. 나이브 베이즈 분류기는 모든 단어가 독립적이라고 가정한다.

    2. 스팸 메일 분류기(Spam Detection)
P(정상메일|입력텍스트)=P(you|정상메일)xP(free|정상메일)xP(lottery|정상메일)xP(정상메일)
P(스팸메일|입력텍스트)=P(you|스팸메일)xP(free|스팸메일)xP(lottery|스팸메일)xP(스팸메일)

예시로 보아(사진참고) P(정상메일)=P(스팸메일)=0.5로 맨 뒤 생략가능.
P(정상메일|입력텍스트)=2/10 x 2/10 x 0/10=0
P(스팸메일|입력텍스트)=2/10 x 3/10 x 2/10=0.012
후자의 확률이 더 크므로 'you free lottery'는 스팸메일로 분류된다.

대충 이런 원리라는 거고 실제로 단어 하나가 0일경우 전체를 0으로 만들어버리는 것은 지나친 일반화이기에
이러한 베이즈정리를 기반으로 하는 나이브 베이즈 분류기는 각 단어 확률 분자,분모에 숫자를 더해 분자가 0이 되는 것을 방지하는 라플라스 스무딩을 사용한다."""
    #3. 뉴스그룹 데이터 분류하기(Classification of 20 News Group with Naive Bayes Classifier)
 #뉴스그룹 데이터에 대한 이해
from sklearn.datasets import fetch_20newsgroups
newsdata=fetch_20newsgroups(subset='train')#all을 넣으면 18,846개 전체 데이터 다운가능하다. train, test로 각기 다른 데이터 다운이 가능하다.
print("뉴스 데이터의 키들: ", newsdata.keys())

print("훈련용 샘플의 개수 확인: data_", len(newsdata.data),", filenames_", len(newsdata.filenames), "target_names_", len(newsdata.target_names), "target_", len(newsdata.target))
print('카테고리: ', newsdata.target_names, '\n')#rec.autos

print('(test)첫번째 샘플의 카테고리: ', newsdata.target[0])
print('(test)첫번째 샘플의 카테고리 이름: ', newsdata.target_names[newsdata.target[0]])
print('(test)첫번째 샘플의 내용: ')
print(newsdata.data[0],'\n')#스포츠 카에 대한 내용

 #나이브 베이즈 분류
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

dtmvector=CountVectorizer()#instantiation
X_train_dtm=dtmvector.fit_transform(newsdata.data)#텍스트를 BoW로 만들어준다.
print("BoW X_train의 모양(shape):", X_train_dtm.shape)

#DTM그대로 나이브 베이즈 분류기에 사용이 가능하지만, TF-IDF가중치를 적용한 TF-IDF Matrix를 입력하면, 성능의 개선을 얻을 수 있다.
#(Frequency만 따져 중요도를 매개버리면 the처럼 의미없는 단어의 weight가 커지기에 TF-IDF가중치를 곱하는데 이는 공통되게 많이 나오면 낮은 weight를, 한 문서에서 자주나오면 높은 weight를 매긴다)
tfidf_transformer=TfidfTransformer()
tfidfv=tfidf_transformer.fit_transform(X_train_dtm)#TF-IDF weight를 적용시켜준다.
print("TF-IDF weight를 적용한 BoW기반 X_train의 모양(shape): ", tfidfv.shape, '\n')

#사이킷런의 나이브베이즈모델
mod=MultinomialNB()#MultinomialNB의 default 인자들은 alpha=1.0, class_prior=None, fit_prior=True이며, alpha는 라플라스 스무딩매개변수, class_prior는 사전확률지정?, fit_prior는 사전확률 학습 여부
mod.fit(tfidfv, newsdata.target)#TF-IDF BoW 텍스트, category

newsdata_test=fetch_20newsgroups(subset='test', shuffle=True)
X_test_dtm=dtmvector.transform(newsdata_test.data)
tfidfv_test=tfidf_transformer.transform(X_test_dtm)

predicted=mod.predict(tfidfv_test)
print('정확도: ', accuracy_score(newsdata_test.target, predicted))#0.77%. (뭐 잠재의미분석챕터전처리를 돌리면 80된다는데 대충 전처리 챕터말하는거같고 당연히 다 적용하면 올라가겠지)
#라플라스 스무딩(Laplace Smooting)은 학습데이터에 없던 값이 들어오거나 이상한 값이 들어올 경우 확률이 0이 되어 전체 확률이 0이 되어버릴 수 있는데, 이를 막기 위해 예전에 한번 있던 값이라고 해버려 0이 아닌 가장 낮은 확률의 값을 주는 것이다.
