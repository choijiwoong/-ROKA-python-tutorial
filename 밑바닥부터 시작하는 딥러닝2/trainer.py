import sys
import time
import numpy
import matplotlib.pyplot as plt
from utils import clip_grads

class Trainer:
    def __init__(self, model, optimizer):
        self.model=model
        self.optimizer=optimizer
        self.loss_list=[]
        self.eval_interval=None
        self.current_epoch=0#갑자기 맥주먹고 싶네

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size=len(x)
        max_iter=data_size//batch_size#배치크기 고려 stop condition of batch calculation
        self.eval_interval=eval_interval
        model, optimizer=self.model, self.optimizer#fitting을 위해 initialization 시 저장한 model, optimizer load
        total_loss=0
        loss_count=0

        start_time=time.time()
        for epoch in range(max_epoch):
            idx=numpy.random.permutation(numpy.arange(data_size))#랜덤으로 뒤섞기
            x=x[idx]#shuffle적용
            t=t[idx]

            for iters in range(max_iter):#배치단위 연산
                batch_x=x[iters*batch_size:(iters+1)*batch_size]#배치생성
                batch_t=t[iters*batch_size:(iters+1)*batch_size]

                loss=model.forward(batch_x, batch_t)#모델 연산 및 손실 계산
                model.backward()#미분
                params, grads=remove_duplicate(model.params, model.grads)#중복되는 값은 쓸모가 없기에 제거
                if max_grad is not None:#clip_grads의 인자가 따로 위에 정의가 돼있다면
                    clip_grads(grads, max_grad)#기울기 정규화
                optimizer.update(params, grads)#기울기 업데이트
                total_loss+=loss#총손실에 가산
                loss_count+=1

                #평가
                if (eval_interval is not None) and (iters%eval_interval)==0:
                    avg_loss=total_loss/loss_count
                    elapsed_time=time.time()-start_time
                    print(' | 에폭 %d | 반복 %d /%d | 시간 %d[s] | 손실 %.2f'%(self.current_epoch+1, iters+1, max_iter, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count=0, 0
                    
            self.current_epoch+=1

    def plot(self, ylim=None):
        x=numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('Iterations (x'+ str(self.eval_interval)+')')
        plt.ylabel('Loss')
        plt.show()

class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model=model
        self.optimizer=optimizer
        self.time_idx=None
        self.ppl_list=None
        self.eval_interval=None
        self.current_epoch=0

    def get_batch(self, x, t, batch_size, time_size):#tiem_size고려하여 x와 t의 batch반환
        batch_x=numpy.empty((batch_size, time_size), dtype='i')
        batch_t=numpy.empty((batch_size, time_size), dtype='i')

        data_size=len(x)
        jump=data_size//batch_size
        offsets=[i*jump for i in range(batch_size)]

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time]=x[(offset+self.time_idx)%data_size]
                batch_t[i, time]=t[(offset+self.time_idx)%data_size]
            self.time_idx+=1
        return batch_x, batch_t

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35, max_grad=None, eval_interval=20):
        data_size=len(xs)
        max_iters=data_size//(batch_size*time_size)
        self.time_idx=0
        self.ppl_list=[]
        self.eval_interval=eval_interval
        model, optimizer=self.model, self.optimizer
        total_loss=0
        loss_count=0

        start_time=time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t=self.get_batch(xs, ts, batch_size, time_size)
                
                #update
                loss=model.forward(batch_x, batch_t)
                model.backward()
                params, grads=remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)#prevention of gradient explosion
                optimizer.update(params, grads)
                total_loss+=loss
                loss_count+=1

                #eval perplexity
                if (eval_interval is not None) and (iters%eval_interval)==0:
                    ppl=numpy.exp(total_loss/loss_count)
                    elapsed_time=time.time()-start_time
                    print("| epoch %d | repeat %d/%d | time %d[s] | perplexity %.2f"%(self.current_epoch+1, iters+1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count=0,0
            self.current_epoch+=1

    def plot(self, ylim=None):#y값의 limit
        x=numpy.arange(len(self.pplt_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('epochs (x'+str(self.eval_interval)+')')
        plt.ylabel('perplexity')
        plt.show()

def remove_duplicate(params, grads):#매개변수 배열 중 중복되는 가중치를 하나로 모아 그에 대응하는 기울기를 더한다. 한마디로 연산 최적화 같음
    params, grads=params[:], grads[:]

    while True:
        find_flg=False
        L=len(params)

        for i in range(0, L-1):#이 과정을 계속 하고, 모든 경우에 대하여 find_flg가 not activated되면 break
            for j in range(i+1, L):#모든 matrix쌍에 대하여
                if params[i] is params[j]:#2개의 params의 값이 같다면(공유한다면)
                    grads[i]+=grads[j]#경사를 더하고.(개인적으로 여기서 경사를 더하고 pop하는게 괜찮은지 의문. 기울기를 반영하여 update를 하는 경우 해당 자리를 기준으로 계산할텐데 그 자리가 없다면 어케되는거지..error안뜨나..)
                    find_flg=True
                    params.pop(j)
                    grads.pop(j)
                elif params[i].ndim==2 and params[j].ndim==2 and params[i].T.shape==params[j].shape and numpy.all(params[i].T==params[j]):
                    grads[i]+=grads[j].T
                    find_flg=True
                    params.pop(j)
                    grads.pop(j)
                if find_flg:
                    break
            if find_flg:
                break
        if not find_flg:
            break
    return params, grads
