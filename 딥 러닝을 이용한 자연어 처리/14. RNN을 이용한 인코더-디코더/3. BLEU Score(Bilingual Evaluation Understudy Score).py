""" 앞선 Language Model의 성능측정방법인 Perplecity는 번역의 성능을 직접적으로 반영하는 수치가 아니기에, 자연어처리에서는 대표적으로 Bilingual Evaluation Understudy를 사용한다.
    [1. BLEU(Bilingual Evaluation Understudy)]
n-gram에 기반하여 기계번역과 사람번역의 유사도를 측정하는 것으로, 완벽하진 않지만 언어가 무엇이든지 빠르게 사용이 가능하다. 그리고 이 방법을 설명하기 이전에 번역성능평가 기준들을 먼저 학습해보자.
사람번역에서 등장한 단어의 개수를 기계번역에서 세어 기게번역 단어개수의 합으로 나누어 평균을 내는 유니그램 정밀도(Unigram Precision)이 있다.
이 Unigram Precision을 향상시키기 이용하여 min(Count, Max_Ref_Count)로 사람번역에 등장한 단어개수를 넘지 못하도록 하여 단어중복을 제거하는데 이를 Modifiec Unigram Precision. 즉 보정된 유니그램 정밀도라고 한다."""
 #Modified Unigram Precision 구현하기
import numpy as np
from collections import Counter
from nltk import ngrams

#Count구현
def simple_count(tokens, n):#tokens에서 n-gram을 카운트
    return Counter(ngrams(tokens, n))

candidate="It is a guide to action which ensures that the military always obeys the commands of the parth."
tokens=candidate.split()
result=simple_count(tokens, 1)
print("유니그램 카운트: ", result)

candidate='the the the the the the the'
tokens=candidate.split()
result=simple_count(tokens, 1)
print('유니그램 카운트: ', result)

#Count_clip구현
def count_clip(candidate, reference_list, n):
    ca_cnt=simple_count(candidate, n)#후보의 simple_count
    max_ref_cnt_dict=dict()

    for ref in reference_list:#답지 리스트(인간번역)
        ref_cnt=simple_count(ref, n)#사람번역의 simple_count

        for n_gram in ref_cnt:#사람번역 각 단어
            if n_gram in max_ref_cnt_dict:#dict에 존재한다면
                max_ref_cnt_dict[n_gram]=max(ref_cnt[n_gram], max_ref_cnt_dict[n_gram])#둘중 최고값으로 넣기
            else:#없으면 그냥
                max_ref_cnt_dict[n_gram]=ref_cnt[n_gram]

    return {n_gram: min(ca_cnt.get(n_gram,0), max_ref_cnt_dict.get(n_gram, 0)) for n_gram in ca_cnt}#후보의 단어별로 min값 을 dict으로 반환

candidate='the the the the the the the'
references=['the cat is on the mat', 'there is a cat on the mat']
result=count_clip(candidate.split(), list(map(lambda ref: ref.split(), references)), 1)
print('보정된 유니그램 카운트: ', result)#기존의 7개카운트에서 2개로 보정됨.

#보정된 정밀도를 연산하는 함수 제작(for 편의)
def modified_precision(candidate, reference_list ,n):
    clip_cnt=count_clip(candidate, reference_list ,n)
    total_clip_cnt=sum(clip_cnt.values())#분자

    cnt=simple_count(candidate, n)
    total_cnt=sum(cnt.values())#분모

    if total_cnt==0:#분모 0방지
        total_cnt=1
    return (total_clip_cnt/total_cnt)
result=modified_precision(candidate.split(), list(map(lambda ref: ref.split(), references)), n=1)
print('보정된 유니그램 정밀도: ', result)

"""이로서 단어중복을 보정하여 유니그램 정밀도를 향상시켰지만, 본질적인 문제 때문에 n-gram으로 확장해야한다.
BoW처럼 빈도수 기반 접근이다보니 단어의 순서가 고려되지 않아 문법이 다 틀린데도 단어빈도만 같으면 정밀도가 동일하게 나온다. 그리하여 순서를 고려하기 위해
n-gram으로 확장을 시키는데, 이 n의 값에 따라 2-gram Precision, 3-gram Precision, 4-gram Precision으로 부르기도 한다.
 여기서 이를 더 향상시키기 위해, BLEU(Bilingual Evaluation Understudy Score)은 n에 따른 모든 n-gram들을 가중치를 사용하여 조합하여 사용한다.
다만 이전에 문제처럼 치명적인 단점이 있었는데, It is같은 짧은 문장의 경우 It kill로 잘못 번역해도 50%의 과한 영향을 끼친다는 것으로 이를 짧은 문장 길이에 대한 패널티 즉, Brevity Penalty라고 한다.
이뿐이 아니라 반대로 기계번역이 너무 길 경우에도 실제 사람이 번역한 문장은 짧아서 1-gram, 2-gram정도만 사용하는데 단지 기계번역 결과가 길다는 이유로
알고리즘 특성상 4-gram같은 걸로 확장해서 불필요한 연산을 진행한다는 문제가 있다.
 고로 기계번역과 사람번역의 길이차이별로 1 혹은 0.xx를 띄는 BP(Brevity Penalty)변수를 기존의 BLEU식에 곱하여 이를 완화시킨다."""
#ref가 여러개라고 가정하여 ca와 길이차이가 가장 작은 ref를 사용하여 BLEU를 계산하자. 최상의 상황 Best match length는 길이가 동일한 것이다.
def closest_ref_length(candidiate, reference_list):
    ca_len=len(candidiate)
    ref_lens=(len(ref) for ref in reference_list)
    closest_ref_len=min(ref_lens, key=lambda ref_len: (abs(ref_len-ca_len), ref_len))#ref_lens를 하나하나 can_len과 비교, 작은값저장
    return closest_ref_len
#BP를 구하는 함수
def brevity_penalty(candidate, reference_list):
    ca_len=len(candidate)
    ref_len=closest_ref_length(candidate, reference_list)

    if ca_len>ref_len:
        return 1
    elif ca_len==0:#비어있다.
        return 0
    else:
        return np.exp(1-ref_len/ca_len)#Brevity_Penalty수치
#위의 함수들을 이용하여 최종적으로 BLEU를 계산하는 함수
def bleu_score(candidate, reference_list, weights=[0.25, 0.25, 0.25, 0.25]):#n=1~4용 weights
    bp=brevity_penalty(candidate, reference_list)
    p_n=[modified_precision(candidate, reference_list, n=n) for n, _ in enumerate(weights, start=1)]#n-gram별 precision
    score=np.sum([w_i*np.log(p_i) if p_i!=0 else 0 for w_i, p_i in zip(weights, p_n)])#가중치와 n-gram별 precision
    return bp*np.exp(score)
#최종적인 BLEU계산을 위해 simple_count, count_clip, modified_precision, bervity_penalty함수를 구현하였다! 근데 사실 BLEU계산 이미 NLTK패키지에 이미 구현되어있다. 허-ㅓㅎ


    #[2. NLTK를 사용한 BLEU측정하기]
import nltk.translate.bleu_score as bleu

candidate='It is a guide to action which ensures that the military always obeys the commands of the parth'
references=[
    'It is a guide to action that ensures that the military will forever heed Party commands',
    'It is the guiding principle which guarantees the military forces always being under the command of the Party',
    'It is the practical guide for the army always to heed the directions of the party'
]

print('실습코드의 BLEU: ', bleu_score(candidate.split(), list(map(lambda ref: ref.split(), references))))
print('NLTK의 BLEU: ', bleu.sentence_bleu(list(map(lambda ref: ref.split(), references)), candidate.split()))#똑같노! 하지만 더 복잡하고 상세하게 구현되어있다고 한다.
