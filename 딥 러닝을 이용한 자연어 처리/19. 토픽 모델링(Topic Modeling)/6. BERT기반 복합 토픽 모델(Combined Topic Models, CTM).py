""" 전통적 빈도수 기반 문서벡터화 방식 Bag of Words와 사전 훈련된 언어모델의 문서 임베딩 방식 SBERT를 결합한 것이 CTM이다.
    1. 문맥을 반영한 토픽 모델(Contextual Topic Models)
문백을 반영한 BERT의 문서 임베딩 표현력과 기존 토픽 모델의 비지도 학습 능력을 결합하여 문서에서 주제를 가져오는 토픽 모델이다.

    2. 데이터 로드하기
https://raw.githubusercontent.com/vinid/data/master/dbpedia_sample_abstract_20k_unprep.txt 에서 다운받은걸
dbpedia_sample_abstract_20k_unprep.txt로 저장한 뒤 text_file변수에 저장한 상황 in colab"""
text_file = "dbpedia_sample_abstract_20k_unprep.txt"

    #3. 전처리
from contextualized_topic_models.models.ctm import CombinedTM#CTM! (Combined Topic Model)
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
import nltk

documents=[line.strip() for line in open(text_file, encoding='utf-8').readlines()]#한줄씩 읽어 저장
sp=WhiteSpacePreprocessing(documents, stopwords_language='english')#불용어처리, 다중띄어쓰기처리에 사용할 WhileSpaceProprocessing인스턴스화
preprocessed_documents, unpreprocessed_corpus, vocab=sp.preprocess()#전처리 적용(vocab_size기본값: 2000). unpreprocessed_corpus까지 저장하는 이유는 SBERT을 사용할 것이기 때문.
print('Bag of words에 사용 될 단어 집합의 크기: ', len(vocab))

tp=TopicModelDataPreparation("paraphrase-distilroberta-base-v1")#Bag of Words와 문맥을 반영한 문서의 BERT임베딩을 얻을 TopicModeldataPreparation인스턴스화
training_dataset=tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)#문맥용과 bow용 따로 fitting
#tp.vocab으로 단어집합 접근 가능

    #4. Combined TM 학습하기
ctm=CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=50, num_epochs=20)#TopicModelDatPreparation을 통한 BoW & Context반영 BERT임베딩 기반 학습
ctm.fit(training_dataset)#CTM자체가 BoW, Context기반으로 임베딩된 벡터라고 생각하고, 이에 대한 학습과 피팅을 진행해준다.(학습 데이터는 위에서 다운받은거)

    #5. 결과 출력
print(ctm.get_topic_lists(5))#각 토픽 별 단어 5개씩 출력. 토픽 개수는 따로 정하는게 없는지 궁그마네..

    #6. 시각화
import pyLDAvis as vis

lda_vis_data=ctm.getldavis_data_format(tp.vocab, training_dataset, n_samples=10)#LDA모델을 가시화데이터로. 단어집합과 데이터를 건네준다.
ctm_pd=vis.prepare(**lda_vis_data)
vis.display(ctm_pd)

    #7. 예측
import numpy as np

topics_predictions=ctm.get_thetas(training_dataset, n_samples=5)#ctm을 사용하여 모든 training_dataset의 토픽예측을 가져온다.
print(preprocessed_documents[0])#전처리 문서의 첫번째 문서 출력
print(ctm.get_topic_lists(5)[0])#첫번째 문서의 토픽 출력

    #8. 모델 저장 및 로드하기
ctm.save(models_dir='./')
del ctm#현재 ctm에 담긴 모델 삭제

ctm=CombinedTM(bow_size=len(tp.vocab), contextual_size=768, num_epochs=100, n_components=50)#세팅 for loading. n_components변수값을 왜 변경..number of topics라는데
ctm.load('/content/contextualized_topic_model_nc_50_tpm_0.0_tpv_0.98_hs_prodLDA_ac_(100, 100)_do_softplus_lr_0.2_mo_0.002_rp_0.99', epoch=19)
print('로드모델 결과: ', ctm.get_topic_lists(5)
