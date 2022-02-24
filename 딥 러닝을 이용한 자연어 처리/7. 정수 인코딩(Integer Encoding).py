#연산효율을 높이기 위해 각 단어를 고유한 정수로 매핑하는 전처리 필요할 경우 정수 인코딩을 사용한다. 보통 단어 등장 빈도수를 기준으로 정렬 뒤, index를 부여한다.

#[dictionary사용하기]
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

raw_text="A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."
sentences=sent_tokenize(raw_text)#sentence segmentation
print("initial sentences: ",sentences, end='\n\n')

#정제, 정규화 병행하며 단어 토큰화_소문자화, 불용어(길이2이하포함)제거. 본격적 자연어 처리 전에 전처리를 최대한 끝내야 한다.
vocab={}#단어의 빈도수를 저장하는 dictionary
preprocessed_sentences=[]
stop_words=set(stopwords.words('english'))#불용어 nltk에서 가져오기

for sentence in sentences:#sentence segmentation이 완료된 sentences변수에서 각 sentence를 get
    tokenized_sentence=word_tokenize(sentence)#tokenization of sentence
    result=[]

    for word in tokenized_sentence:#모든 토큰들에 대하여
        word=word.lower()#소문자화를 시킨 뒤에

        if word not in stop_words:#불용어에 포함되지 않으며
            if len(word)>2:#길이가 2 이하가 아니라면
                result.append(word)#해당 단어를 result에 append한 뒤에

                if word not in vocab:#해당 단어가 vocab에 있지 않는 단어라면
                    vocab[word]=0#단어의 빈도수를 0으로 초기화
                vocab[word]+=1#일괄적으로 1을 더해 초기화 된 값을 1로, 이미 있는단어라면 +1을.
    preprocessed_sentences.append(result)#해당 result값을 preprocessed_sentences변수에 append한다. 우리가 초기에 넣은 값은 여러 문장이니.
print('preprocessed_sentences: ', preprocessed_sentences, end='\n\n')#솔직히 다른 단어로 통합하는 정규화는 사용되지 않은듯.
print('how many?: ', vocab, end='\n\n')
print("frequency of barber: ", vocab["barber"], end='\n\n')

vocab_sorted=sorted(vocab.items(), key=lambda x:x[1], reverse=True)#vocab의 데이터를 해당 데이터의 index1의 값 기준(키), 역순으로 정렬
print("sorted vocab by frequency: ", vocab_sorted, end='\n\n')

#빈도순위를 매기기 위해 빈도수가 높으면 낮은 정수 부여
word_to_index={}
i=0
for (word, frequency) in vocab_sorted:#정렬된 vocab의 단어와 빈도수를 가져와,
    if(frequency>1):#(빈도수가 적은(1개인) 단어는 제외하고 순위책정). 사실 이것이 이번 주제의 핵심인게, 불필요한 값을 지속적으로 줄여가는 과정.
        i=i+1#등수를 올린 뒤(초기값이 0이기에)_일괄적인 등수책정
        word_to_index[word]=i#해당 값을 word_to_index[word]에 저장.
print("ranking in vocab_sorted: ", word_to_index, end='\n\n')

#불필요한 값을 더 줄이기 위해 실용적인 상위 5개값만 사용한다면
vocab_size=5
words_frequency=[word for word, index in word_to_index.items() if index>=vocab_size+1]#word_to_index(랭킹된)아이템 중, 순위가 vocab_size(5) 초과라면 해당 word를 저장
#즉, 순위 6이상의 값을 words_frequency에 저장

for w in words_frequency:#순위 6이상의 값들을
    del word_to_index[w]#word_to_index dictionary에서 삭제
print("순위 5이하의 값들만 저장되어있는 word_to_index(순위매겨진 리스트): ", word_to_index, end='\n\n')#단어를 정수로 인코딩_원활한 처리를 위하여(순위5이하)

#하지만 여기서 word_to_index에서 존재하지 않는 단어는(불필요하다 생각되어 처리된_ex. 불용어, 빈도낮음) 따로 Out-Of-Vocabulary(OOV)문제라고 하는데,
#이는 word_to_index에 'OOV'라는 단어를 추가하고 이 인덱스로 인코딩하여 처리한다.
word_to_index['OOV']=len(word_to_index)+1#왜 초기값을 word_to_index의 크기로 하지?*******A. 그냥 둘데 없는데 차례대로 두려고. 이 OOV index는 따로 문서화해두는 게 좋을 듯.
print("appended OOV to word_to_index: ", word_to_index, end='\n\n')

#word_to_index 즉, tokenized된 것 중 정제를 거쳐 등장 빈도수가 높은 것으로만 이루어진 word_to_index를 이용하여 sentences의 모든 token들을 정수로 매핑. 인코딩한다.
encoded_sentences=[]
for sentence in preprocessed_sentences:#전처리된 sentences들에 대하여 각 sentence들을
    encoded_sentence=[]
    for word in sentence:#각 sentence의 word들을
        try:
            encoded_sentence.append(word_to_index[word])#word_to_index기반하여 빈도랭킹 정수를 encoded_sentence에 append한다.
        except KeyError:#만약 해당 word의 키값(랭크값)이 word_to_index에 없다면 KeyError가 일어날 것이고,
            encoded_sentence.append(word_to_index['OOV'])#OOV 로 간주하여 OOV의 정수를 append한다.(이렇게 알고리즘에 except를 활용해도 되나...)
    encoded_sentences.append(encoded_sentence)#해당 결과를 encoded_sentences에 append한다.
print("preprocessed_sentence를 word_to_index를 이용하여 mapping한 결과(encoded_sentences): ", encoded_sentences)
            

#이러한 정수 인코딩과정을 지원하는 모듈로 Counter, FreqDist, enumerate, Keras_tokenizer가 있다.
