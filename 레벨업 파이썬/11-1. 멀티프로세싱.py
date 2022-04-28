#[메인 프로세스]
import multiprocessing as mp

if __name__=="__main__":
    proc=mp.current_process()#현재 실행중인 프로세스의 정보를 가져와서
    print(proc.name)#이름와 Process ID출력
    print(proc.pid)

#[프로세스 스포닝(spawning)_Parent Process가 Child Process를 새로 만들어 내는 과정. 일부 작업을 위임한다]
def worker():
    print("SubProcess End")#왜 안뜨징..

if __name__=="__main__":
    p=mp.Process(name='SubProcess', target=worker)#프로세스 스포닝. 
    p.start()#자식 프로세스를 가동시킨다.
    

import time#각 프로세스의 이름과 PID를 출력하도록 변경해보자

def worker():
    proc=mp.current_process()
    print(proc.name)
    print(proc.pid)
    time.sleep(5)
    print("SubProcess End")#왜 안뜨징..

if __name__=='__main__':
    proc=mp.current_process()
    print(proc.name)
    print(proc.pid)

    p=mp.Process(name='SubProcess', target=worker)
    p.start()

    print("MainProcess End")


#[Pool 사용하기]
from multiprocessing import Pool

def work(x):
    print(x)

if __name__=='__main__':
    pool=Pool(4)#대충 4개 쓰레드? 생성하고
    data=range(1,100)#데이터를
    pool.map(work, data)#pool들에게 work시키는듯. 만약 thread별로 다른 작업을 시키고 싶다면 work단에서 pid체크하면 될듯
