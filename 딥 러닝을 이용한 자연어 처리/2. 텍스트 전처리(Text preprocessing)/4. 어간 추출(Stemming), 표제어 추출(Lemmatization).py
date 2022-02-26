"""
[어간추출과 표제어 추출]
코퍼스의 단어 개수를 줄일 수 있는 기법들로 lemmatization과 stemming이 있으며, 의미도 다른 단어지만 하나의 단어로 일반화 시킬 수있을 경우 줄이는 것이다.
이는 단어의 빈도수를 기반으로 문제를 푸는 BoW(Bag of Words)표현의 자연어처리 문제에 주로 사용된다.

[표제어추출]
이는 기본 사전형 단어의 의미로 뿌리 단어를 찾아가는 것이다. is, are, am의 표제어는 be이다.
우선 형태소는 어간(stem)_핵심과 접사(affix)_추가의미로 이루어지는데 형태학적 파싱은 이 두 요소를 분리하는 작업을 말한다. cats=cat+s
 이를 위해 NLTK에서 표제어추출 도구인 WordNetLemmatizer을 지원한다.
"""
from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()

words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

print("표제어 추출 전: ", words)
print("표제어 추출 후: ", [lemmatizer.lemmatize(word) for word in words])#comprehension!
#dy, ha, watchec

#표제어 추출은 어간 추출과는 달리 단어의 형태가 적절히 보존되나, 의미를 알 수 없는 단어도 일부 출력하는데, lemmatizer가 본래 단어의 품사 정보를 알아야 정확한 결과를 얻을 수 있기 때문이다.
#고로 WordNetLemmatizer는 단어의 품사정보를 전달해줄 수 있다.
print("\nlemmatizer에 품사정보를 갖이 전달:", lemmatizer.lemmatize('dies', 'v'), end=', ')
print(lemmatizer.lemmatize('watched', 'v'), end=', ')
print(lemmatizer.lemmatize('has', 'v'))


#[어간 추출]
#이는 형태학적 분석의 단순화버전으로 어림짐작하기에 사전에 존재하지 않는 단어일 수도 있다. Porter Algorithm의 예시이다.

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

porter_stemmer=PorterStemmer()

sentence="This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
tokenized_sentence=word_tokenize(sentence)#먼저 토큰화 한 뒤

print('\n어간 추출 전: ', tokenized_sentence)
print('어간 추출 후 with PorterStemmer: ', [porter_stemmer.stem(word) for word in tokenized_sentence])#stemmer에게 전달하여 어간추출
#규칙기반의 접근을 하므로 사전에 없는 단어들도 포함될 수 있다. 규칙예시: ALIZE->AL, ANCE->삭제, ICAL->IC

words=['formalize', 'allowance', 'electricical']

print("어간 추출 전: ", words)
print("어간 추출 후 with PorterStemmer: ", [porter_stemmer.stem(word) for word in words])#위와 같은 어간추출기 PorterStemmer규칙에 따르면 위 words는 잘 분류된다.
#(여기까지 보고 내가 느낀건 표제어 추출보다는 어간 추출이 좀 더 나을거같다는 거.. )
#어간 추출의 속도는 표제어 추출보다 일반적으로 빠르며, NLTK에서는 포터 알고리즘 외에도 Lancaster Stemmer알고리즘을 지원한다. 결과를 비교해보자.

from nltk.stem import LancasterStemmer

lancaster_stemmer=LancasterStemmer()
words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print("\n어간 추출 전: ", words)
print("포터 스테머의 어간추출 후: ", [porter_stemmer.stem(w) for w in words])
print("랭커스터 스태머의 어간추출 후: ", [lancaster_stemmer.stem(w) for w in words])
"""
둘이 전혀 다른 결과를 보여주기에 사용하고자 하는 코퍼스에서 어떤 스태머가 적합한지 판단후 사용해야한다. 잘못 어간추출된 단어가 실제로 그렇게 쓰인 단어와 같다고 판단한다면
의미가 동일한 겨웅에만 같은 단어로 취급하는 정규화의 목적에 반대되기 때문이다.
Stemming_am->am, the going->the go, having->hav
Lemmatization_am->be, the going->the going, having->have

여기서 개인적으로 표제어추출과 어간추출의 차이점이 모호해지는데, 표제어 추출은 기본 사전형 단어, 즉 다른 형태를 가지더라도 그 뿌리단어를 찾아가서
단어의 개수를 줄일 수 있는지를 판단하는 과정이고(고로 사전적 단어가 나옴),
어간추출은 뿌리로 올라가며 찾는 것이 아니라 어림짐작하여 찾는 것으로 사전적 단어가 나오지 않을 수 있다.
 즉, 둘은 모두 형태학적 분석이며, 방법이 다른 것이다. 이둘은 모두 정규화 기법 중에 단어의 개수를 줄이기 위해 사용되는 기법이다.

[한국어에서의 어간 추출]
한국어는 5언(체언, 수식언, 관계언, 독립언, 용언) 9품사(명사, 대명사, 수사/ 관형사, 부사/ 조사/ 감탄사/ 동사, 형용사)의 구조를 가진다.

인도유럽어, 한국어의 통칭적 개념인 활용(Conjugation)은 어간(stem)이 어미(ending)을 가지는 일을 뜻한다.
이 어간이 어미를 취하는 과정에서, 어간의 모습이 일정하면 규칙활용, 어간이나 어미의 모습이 변하면 불규칙 활용으로 분류한다.

규칙활용은 잡/다, 잡/어, 잡/기로 처럼 붙기전과 붙은 후의 모습이 같기에 어미만 분류해주면 어간추출(stemming)이 된다.
반면에, 불규칙 활용은 듣다&들었다, 곱다&고우시다, 혹은 특수한 어미를 취하는 경우 푸르+아/어->푸르어, 이르+아/어->이르러 가 해당한다.
 고로 불규칙 활용의 경우 보다 복잡한 규칙을 필요로 한다.
"""
