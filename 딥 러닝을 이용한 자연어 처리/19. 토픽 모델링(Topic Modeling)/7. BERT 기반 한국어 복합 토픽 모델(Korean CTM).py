from contextualized_topic_models.ctm import CombinedTM#CTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation, bert_embeddings_from_list#BoW+Context데이터, ?
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing#preprocessing tool
from sklearn.feature_extraction.text import CountVectorizer#BoW
from konlpy.tag import Mecab#morphs
from tqdm import tqdm#process bar

#Preprocessing
documents=[line.strip() for line in open(text_file, encoding='utf-8').realines()]

preprocessed_documents=[]
for line in tqdm(documents):
    if line and not line.replace(' ', '').isdecimal():#빈 문자열, 숫자 제외
        preprocessed_documents.append(line)
print('preprocessed_documents의 길이: ', len(preprocessed_documents))

#Tokenizer
class CustomTokenizer:#Instantiation시 인자로 tagger를 받아 token화하여 단어별로 리스트 저장(길이 1초과)
    def __init__(self, tagger):
        self.tagger=tagger

    def __call__(self, sent):
        word_tokens=self.tagger.morphs(sent)
        result=[word for word in word_tokens if len(word)>1]
        return result
custom_tokenizer=CustomTokenizer(Mecab())#tagger로 Mecab전달. CounterVectorizer자체적으로 sklearn의 띄어쓰기 토큰화를 진행하지만, 한국어에는 적절치 않다.

#Vectorizer
vectorizer=CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)#For BoW
train_bow_embeddings=vectorizer.fit_transform(preprocessed_documents)#CountVectorizer로 preprocessed_documents 피팅(BoW 임베딩)
print('train_bow_embeddings크기: ', train_bow_embeddings.shape)#(27540, 3000)

#vocab, id2token
vocab=vectorizer.get_feature_names()#빈도수 기반 vocab
id2token={k:v for k, v in zip(range(0, len(vocab)), vocab)}#for integer indexing?
print('vocab size: ', len(vocab))

#contextualized_embedding(BERT모델을 이용한 context embedding)
train_contextualized_embeddings=bert_embeddings_from_list(preprocessed_documents, 'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

#TopicModel
qt=TopicModelDataPreparation()#topicmodel 데이터 준비(context와 bow을 전달)
training_dataset=qt.load(train_contextualized_embeddings, train_bow_embeddings, id2token)#TopicModel에 context데이터, bow데이터 입력, id2token은 왜 입력하는지 잘 모르겠네. vocab을 추가적으로 전달해주는건가

#Training
ctm=CombinedTM(bow_size=len(vocab), contextual_size=768, n_components=50, num_epochs=20)
ctm.fit(training_dataset)#context와 bow가 합쳐진 임베딩을 CombinedTM을 통해 전달.

print('토픽들: ', ctm.get_topics(10))

#Visualization
import pyLDAvis as vis

lda_vis_data=ctm.get_ldavis_data_format(vocab, training_dataset, n_samples=10)

ctm_pd=vis.prepare(**lda_vis_data)
vis.display(ctm_pd)
