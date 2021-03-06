""" 학습 기반 tokenizer는 OOV(Out-Of-Vocabulary), UNK(Unknown Token)의 한계가 있다. 이걸 OOV문제라고 하는데,
Subword Segmentaion으로 이를 어느정도 완화하는데, 이런 작업을 하는 Tokenizer를 Subword Tokenzier라고 한다.
주요 알고리즘으로 바이트 페어 인코딩과 SentencePiece, Huggingface의 Tokenizers가 있다

    [1. BPE(Byte Pair Encoding)]
데이터 압축 알고리즘으로 aaabdaaabac처럼 연속적으로 많이 등장한 글자쌍을 찾아 하나의 글자로 치환하며, 여기서의 글자는 byte에 비유된다.
Z=aa, Y=ab, X=ZY일때 위 문장은 XdXac가 된다.

    [2. 자연어 처리에서의 CPE(Byte Pair Encoding)]
 1. 기존의 접근
빈도수 카운트 데이터 low:5, lower:2, newest:6, widset:3(등장 빈도수)를 예로, 이 훈련 데이터의 vocab은 low, lower, newest, widest이다.
여기에 lowest를 넣으면 OOV문제가 발생한다. 이게 기존의 접근이다.

 2.BPE 알고리즘을 사용한 경우
문자단위로 분리한다. l o w:5, l o w e r:2, n e w e s t:6, w i d s e t:3 이 데이터의 vocab은 문자단위로 분리된 l o w e r n w s t i d이다.
BPE의 특징은 알고리즘의 동작을 몇회 반복(iterate)할 것인지 사용자가 정하는데, 10회로 가정해보자. 빈도수가 높은 유니그램의 쌍을 하나의 유니그램으로 통합하는 과정을 10회 반복하는 것이다.
 빈도수가 높은 쌍을 통합한다. 1회: (e,s) 2회: (es, t) 3회: (l,o) 4회:... 10회의 vocab과 단어 집합은 아래와 같다.
low:5, low e r:2, newest:6, widest:3 / vocab: l o w e r n w s t i d es est lo low ne new newest wi wid widest
 이제는 'lowest'가 등장하면 OOV가 아니라 글자단위로 분리하여 l o w e s t를 찾아내서 low와 est를 찾아내어 이로 인코딩한다. OOV가 아니다!"""
#3. 코드 실습하기
import re, collections
from IPython.display import display, Markdown, Latex

num_merges=10#iteration of BPE

dictionary = {'l o w </w>' : 5,
         'l o w e r </w>' : 2,
         'n e w e s t </w>':6,
         'w i d e s t </w>':3
         }

#빈도수가 높은 유니그램 쌍을 하나의 유니그램으로 통합하는 과정
def get_stats(dictionary):#유니그래의 pair들의 빈도수를 카운트
    pairs=collections.defaultdict(int)#dictionary와 비슷하지만, key값이 없을 경우 미리 지정해둔 default값을 반환하는 dictionary이다. 현재의 경우 int 0반환
    for word, freq in dictionary.items():
        symbols=word.split()
        for i in range(len(symbols)-1):#thanks to <\w>
            pairs[symbols[i], symbols[i+1]]+=freq#현재 연속된 pair들이 큰 단어만큼 반복된것이기에 단어의 freq를 더한다.
    print('현재 pair들의 빈도수: ', dict(pairs))
    return pairs

def merge_dictionary(pair, v_in):#아마 정황상 횟수에 따라 가장 빈도수가 높은 pair를 병합한다는 거겠지? 설명 좆같이 없네
    v_out={}#(pair가 merge되어 반환될 dictionary)
    bigram=re.escape(' '.join(pair))#패턴을 입력받으면 이스케이프(백슬래쉬)로 바꾼 뒤 반환한다. 들어온 pair에서 특수문자를 제거한다!(둘 사이 공백을 제거하고 붙인다)
    p=re.compile(r'(?<!\S)'+bigram+r'(?!\S)')#특수문자+문자+특수문자 regex패턴을 컴파일한다.(그걸 pattern으로서 사용)
    for word in v_in:
        w_out=p.sub(''.join(pair), word)#v_in의 단어가 위의 regex패턴을 만족한다면(_1로 치환한다. _2의 데이터속에서)
        v_out[w_out]=v_in[word]#그걸 key삼아 단어를 저장한다.
    return v_out
"""아. 그 BPE의 입력이 위의 dictionary꼴이니까 단어의 끝(<\w>)을 구분하기 위해 regex pattern을 compile해둔거고 v_in의 각 토큰들에서
word의 내용을 보아 특수문자패턴이 나오면 그걸 pair의 내용으로 바꾼다. 그리고 그 결과값을 v_out의 key로 그 단어를 저장한다."""#*******
bpe_codes={}
bpe_codes_reverse={}

for i in range(num_merges):
    display(Markdown('### Iteration {}'.format(i+1)))
    pairs=get_stats(dictionary)#pair들의 빈도수가 담긴 dictionary
    best=max(pairs, key=pairs.get)#빈도수가장 높은 key를 획득
    dictionary=merge_dictionary(best, dictionary)#대충 해당 best pair를 merge하여 dictionary를 반환하는듯. 지금 새롭게 다는 주석을 ()로 하겠음. 근데 메카니즘을 모르겠네ㅎㅎ

    bpe_codes[best]=i
    bpe_codes_reverse[best[0]+best[1]]=best

    print('new merge:', best)
    print('dictionary: ', dictionary)
print('\n\nmerge의 기록 출력: ', bpe_codes,'\n')


#4. OOV에 대처하기
def get_pairs(word):#그냥 word의 pair들의 set을 반환
    pairs=set()
    prev_char=word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char=char
    return pairs

def encode(orig):#단어가  들어가려나? ㅇㅇ
    word=tuple(orig)+('</w>',)
    print("__word split into characters:__ <tt>{}</tt>".format(word))

    pairs=get_pairs(word)

    if not pairs:#empty
        return orig

    iteration=0
    while True:
        iteration+=1
        print("__Iteration {}:__".format(iteration))#몇번째 iteration인지 출력
        print('bigrams in the word: ', pairs)#단어 안에 있는 bigram(pair)를 출력

        bigram=min(pairs, key=lambda pairs: bpe_codes.get(pairs, float('inf')))#bpe_codes(merge log)에서 가장 최근에 된 값(0부터 indexing되니)
        print('candidate for merging:' ,bigram)

        if bigram not in bpe_codes:#만약 여기에 없다면(thanks to defautdict, 위에서 0출력하고 오류없음)
            display(Markdown("__Candidate not in CPE merges, algorithm stops.__"))
            break

        first, second=bigram
        new_word=[]
        i=0
        while i<len(word):#char별 반복
            try:
                j=word.index(first, i)#first가 word에 어디index에 위치하는지
                new_word.extend(word[i:j])#new_word에 i(현재char)~(first위치)를 append
                i=j#i조작
            except:
                new_word.extend(word[i:])#끝에 달했다면 나머지 그냥 append
                break

            if word[i]==first and i<len(word)-1 and word[i+1]==second:#모든게 완벽하다면? 뭔가 위에 try부분이 well compile시
                new_word.append(first+second)#merge?하여 append하고
                i+=2
            else:#뭔가 except시
                new_word.append(word[i])
                i+=1
        new_word=tuple(new_word)#tuple화
        word=new_word#대체
        print('word after merging', word)
        if len(word)==1:
            break
        else:
            pairs=get_pairs(word)

    if word[-1]=='</w>':#for 출력X
        word=word[:-1]
    elif word[-1].endswith('</w>'):#만약 </w> 접미사로 끝나면
        word=word[:-1]+(word[-1].replace('</w>',''),)#접미사 변경

    return word
#단어를 bigrams로 만든다음에 get_pair을 통해 빈도수 기반 merging을 수행할 pair를 결정한다. 그 후 merge라고 변환된 단어를 리턴한다.
encode("loki")#서브워드 단어 집합에 lo가 존재하기에 lo를 유지, 나머지 분리
encode('lowest')#서브워드에 low와 est가 존재, 이들을 분리
encode('lowing')#서브워드에 low가 존재 
encode('highing')#서브워드에도 안존재. 모든 char가 분리.

#씨발 한도제한 계좌ㅗ 시간을 얼마나 버린거야
    """[3. WordPiece Tokenizer]
BPE의 변형 알고리즘으로 BPE가 빈도수 기반으로 쌍을 선택해 병합하는 것과 달리, 병합되었을때의 코퍼스의 우도(Likelihoos)를 가장 높이는 쌍을 병합한다
WordPiece Tokenizer는 모든 단어의 맨 앞에 _를 붙이고, 단어는 서브워드 통계에 기반하여 띄어쓰기로 분리한다. _는 문장 복원을 위한 장치이다
BERT를 훈련시키기위해서도 사용된다.

    [4.Unigram Language Model Tokenizer]
각 서브워드들의 loss를 계산하는 것으로, lsos는 서브워드가 vocab에서 제거되었을때 코퍼스의 우도(Liktlihood)가 감소하는 정도를 말한다.
서브워드들을 loss기준 정렬하여 최악의 10~20%의 토큰을 제거하여 vocab크기를 맞출때까지 반복한다."""
