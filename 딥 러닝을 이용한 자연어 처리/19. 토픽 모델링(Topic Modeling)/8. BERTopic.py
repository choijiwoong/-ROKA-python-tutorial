"""LDA 대체가능 기술이라는 썰.
    1. BERTopic
BERT contextual embedding과 class-based TF-IDF를 활용하여 중요한 단어를 유지하며 쉽게 해석할 수 있는 클러스터를 만드는 토픽 모델링 기술이다.
우선 SBERT로 임베딩 하고 UMAP을 이용하여 임베딩 차원을 줄인 후 HDBSCAN으로 축소된 임베딩을 클러스터링하고 의미적으로 유사한 문서 클러스터를 생성한다.
그 뒤 클래스 기반 TF-IDF로 토픽을 추출한다.
 UMAP(Uniform Manifold Approximation and Projection)은 topological data분석으로 manifold learning기술을 기반으로 한 차원 축소로 그냥 빠르고 더 명확하게(t-SNE보다) 카테고리 분리를 한다.
Manifold Learning의 Hypothesis는 TrainingDB 데이터를 공간상에 나타냈을 때, DB의 데이터를 잘 아우르는 subspace가 존재할 것이다 이다. 이러한 매니폴드를 찾는 것이 매니폴드 학습이며 사진을 참고하자.
 또한 일반 TF-IDF와 c-TF-TDF는 우선 모든 문서에 동일한 클래스 벡터를 제공하기 위함으로, 개개인이 아닌 모든 문서를 함께 결합한 것을 사용하는 아이디어이다.
 문서에 TF-IDF를 적용하는 것이 아닌 문서를 병합하고 TF-IDF에서 문서 수가 아닌 클래스 수를 사용한다. 이때 상황에 맞게 클래스 수를 조절하며 클래스 수의 의미는 나도 모른다
기존 Density Based Clustering의 대표적인 DBSCAN방법론의 local density 정보 미반영, 계층구조 미반영의 단점을 보완한 것이 HDBSCAN으로 각 군집 내 데이터가 최소한의 밀도를 만족시키게 하는 밀도 기반 기법으로 non-curcle 임의형태군집발견에 효과적이다"""
    #2. 데이터 로드
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs=fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
print('총 문서의 개수: ', len(docs))#18,846

    #3. 토픽 모델링
model=BERTopic()
topics, probabilities=model.fit_transform(docs)#문서 피팅
print('각 문서의 토픽 번호 리스트: ', len(topics))#18,846
print('첫번째 문서의 토픽 번호: ', topics[0])#현재 토픽은 211개 존재.

model.get_topic_info()#(get_topic_info이용 토픽수, 토픽크기, 단어 일부 show). Topic -1은 토픽이 할당되지 않은 이상치 문서(outliers)를 나타낸다.
print('총 문서의 수: ', model.get_topic_info()['Count'].sum())
print('임의의 토픽 단어들: ', model.get_topic(5))#5번 토픽 단어 출력

    #4. 토픽 시각화
model.visualize_topics()#LDAvis와 유사한 방식의 시각화 지원.

    #5. 단어 시각화
model.visualization_batchart()#c-IF-IDF점수에서 막대 차트로 각 토픽 단어들 비교 가능

    #6. 토픽 유사도 시각화
model.visualize_heatmap()#마우스를 대어 실질적 유사도 확인

    #7. 토픽의 수 정하기(제한)
model=BERTopic(nr_topics=20)#현재 토픽 20개만 존재
topics, probabilities=model.fit_transform(docs)

model.visualize_topics()#20개 토픽만 가시화


model=BERTopic(nr_topics='auto')#자동으로 토픽 수 설정. 현재 토픽 144개 존재
topics, probabilities=model.fit_transform(docs)

model.get_topic_info()

    #8. 임의의 문서에 대한 예측
new_dox=docs[0]
print(new_doc)

topics, probs=model.transform([new_doc])
print('예측한 토픽 번호: ', topics)#사용법 존나게 간단하넴

    #9. 모델 저장과 로드
model.save('my_topics_model')
BerTopic_model=BERTopic.load('my_topics_model')

#한국어 모델은 다국어BERT, mecab이용하는것이 차이다.
