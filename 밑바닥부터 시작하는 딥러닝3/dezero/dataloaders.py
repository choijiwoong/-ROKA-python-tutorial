import math
import random
import numpy as np
from dezero import cuda

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        self.dataset=dataset
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.data_size=len(dataset)
        self.max_iter=math.ceil(self.data_size/batch_size)
        self.gpu=gpu

        self.reset()

    def reset(self):
        self.iteration=0
        if self.shuffle:
            self.index=np.random.permutation(len(self.dataset))
        else:#그렇지 않을경우 그냥 순서대로
            self.index=np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration>=self.max_iter:
            self.reset()
            raise StopIteration()

        i, batch_size=self.iteration, self.batch_size
        batch_index=self.index[i*batch_size:(i+1)*batch_size]
        batch=[self.dataset[i] for i in batch_index]
        
        xp=cuda.cupy if self.gpu else np#GPU지원
        x=xp.array([example[0] for example in batch])
        t=xp.array([example[1] for example in batch])

        self.iteration+=1
        return x, t#ndarray로

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu=False

    def to_gpu(self):
        self.gpu=True

    def __len__(self):
        return self.data_size

class SeqDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, gpu=False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False, gpu=gpu)

    def __next__(self):
        if self.iteration>=self.max_iter:
            self.reset()
            raise StopIteration

        jump=self.data_size//self.batch_size
        batch_index=[(i*jump+self.iteration)%self.data_size for i in range(self.batch_size)]#배치좀 특이한데 jump위치별로 하나씩 뽑아 batch_size만큼되게하기.
        batch=[self.dataset[i] for i in batch_index]

        xp=cuda.cupy if self.gpu else np
        x=xp.array([example[0] for example in batch])
        t=xp.array([example[1] for example in batch])

        self.iteration+=1
        return x, t
