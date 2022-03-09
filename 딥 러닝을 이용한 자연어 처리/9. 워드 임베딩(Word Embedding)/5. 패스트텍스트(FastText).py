'''페이스북의 단어를 벡터화시키는 방법으로, Word2Vec의 확장으로 볼 수 있으며 차이점은 하나의 단어 안에도 여러 단어들(subword)를 고려한다는 것이다.
    1. 내부 단어(subword)의 학습
각 단어를 글자단위 n-gram으로 취급하며, <>을 도입하여 subword토큰을 벡터로 만든다. ex) n=3이면 <ap,app,ppl,ple,le>, <apple>을 벡터화(6개토큰)
n은 3~6에서 설정이 가능하다. 이러한 단어들(subword)에 대하여 Word2Vec을 수행하고, 그 합으로 apple의 벡터값을 구성한다.

    2. 모르는 단어(Out Of Vocabulary, OOV)에 대한 대응
FastText는 데이서셋의 모든 단어의 n-gram에 대해 워드 임베딩되는데 이의 장점은 데이터셋이 충분하다면 subword로 out of vocabulary에 대해
다른 단어와의 유사도를 계산할 수 있다는 것이다. birth, place의 벡터가 있다면 birthplace에 대한 vector를 얻을 수 있다는 것이다.

    3. 단어 집합 내 빈도 수가 적었던 단어(Rare word)에 대한 대응
FastText의 경우 Word2Vec와 달리, 단어가 희귀단어여도 n-gram이 다른 단어의 n-gram과 겹치면 비교적 높은 임베딩 벡터값을 얻는다.
이는 오타의 경우 진가를 발휘하는데, Word2Vec은 희귀단어로 취급하지만, FastText는 오타의 원래 단어의 n-gram과 겹쳐져 일정수준의 성능을 보인다(appple와 apple의 n-gram동일)

    4. 실습으로 비교하는 Word2Vec vx FastText(Word2Vec학습코드와 전처리코드를 그대로 수행했다고 가정.'''
#1. Word2Vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

#model=Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, ag=0)
loaded_model=KeyedVectors.load_word2vec_format("eng_w2v")
try:
    print("electrofishiing의 유사한 단어(Word2Vec): ", loaded_model.most_similar("electrohishing"))#에러 발생! 존재하지않는단어
except KeyError as e:
    print("Key가 vocab에 존재하지 않습니다! ", e)


#2. FastText
from gensim.models import FastText

model=FastText(result, size=100, window=5, min_count=5, workers=4, sg=1)
print("eletrofishing의 유사한 단어(FastText): ", model.sv.most_similar("eletrohishing"))
#[('electrolux', 0.7934642434120178), ('electrolyte', 0.78279709815979), ('electro', 0.779127836227417),
#('electric', 0.7753111720085144), ('airbus', 0.7648627758026123), ('fukushima', 0.7612422704696655),
#('electrochemical', 0.7611693143844604), ('gastric', 0.7483425140380859), ('electroshock', 0.7477173805236816),
#('overfishing', 0.7435552477836609)]


"""3. 한국어에서의 FastText
"자연어처리" 음절단위의 n-gram(n=3)은 <자연, 자연어, 연어처, 어처리, 처리>이다.
오타나 노이즈 측면에서 강한 임베딩을 위해 자모 단위로 임베딩한다면 <ㅈㅏ, ㅈㅏ_, ㅏ_ㅇ, _ㅇㅕ, ㅇㅕㄴ,ㅕㄴㅇ, ..., _ㄹㅣ, ㄹㅣ>가 된다. 빈 종성은_로."""
