#[멀티프로세스와 큐]
import multiprocessing as mp

q=mp.Queue()#멀티프로세스 전용 Queue인가.. 근데 일반적인 Queue에 유용메서드 몇개 추가된것일듯.
q.put(1)
q.put(2)
q.put(3)

data1=q.get()
data2=q.get()
data3=q.get()

print(data1)
print(data2)
print(data3)


#[멀티프로세싱과 PyQt]
"""GUI에서 오랜작업을 할 시 멈추는 현상이 발생할 수 있기에 별도의 스레드를 생성하는 것이 좋다. 만약 실제 CPU를 사용하는 복잡한 계산이라면 프로세스를 사용하는게 좋다.
파이썬은 GIL(Global Interpeter Lock)의 제한으로 멀티 스레드의 성능이 더 좋지않기에 멀티프로세스가 권장된다.(계산많을시)"""
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from multiprocessing import Process, Queue
import multiprocessing as mp
import datetime
import time

def producer(q):#프로세스 이름을 출력하고, 현재의 시간데이터를 1초 간격으로 queue에 put한다.
    proc=mp.current_process()
    print(proc.name)

    while True:
        now=datatime.datetime.now()
        data=str(now)
        q.put(data)
        time.sleep(1)

class Consumer(QThread):
    poped=pyqtSignal(str)#사용자 정의 시그널을 생성, 특정 이벤트가 발생했을 때 시그널을 방출되게 할 수 있다.

    def __init__(self, q):
        super().__init__()
        self.q=q

    def run(self):
        while True:
            if not self.q.empty():#queue에 producer가 put하여 데이터가 존재한다면
                data=q.get()#데이터(str타입 시간)를 가져온 뒤
                self.poped.emit(data)#시그널을 방출시킨다. 이 시그널은 아래 MyWindow에서 사용할 예정이다.

class MyWindow(QMainWindow):#Window창에 대한 속성 상속
    def __init__(self, q):
        super().__init__()
        self.setGeometry(200, 200, 300, 200)#크기 설정

        self.consumer=Consumer(q)#Consumer를 설정하고
        self.consumer.poped.connect(self.print_data)#consumer에서 poped되는 시그널을 self.print_data즉 QMainWindow에 출력할 데이터에 연결시킨다.
        self.consumer.start()#가동

    @pyqtSlot(str)#데코레이터.
    def print_data(self, data):
        self.statusBar().showMesage(data)#print_data로 들어간 시간 데이터가 담긴 시그널을 showMessage로 윈도우에 출력시킨다.

if __name__=='__main__':
    q=Queue()#큐를 생성하고

    p=Process(name='producer', target=producer, args=(q,), daemon=True)#producer을 child process로 만드는데, 인자로 queue를 넣는다.
    p.start()#producer가 별도의 process에서 queue에 put하는 작업 시작

    app=QApplication(sys.argv)
    mywindow=MyWindow(q)
    mywindow.show()
    app.exec_()#ㅠㅠ 잘 안댐


#[멀티 프로세싱과 클래스]
import multiprocessing as mp

class Worker:
    def __init__(self):
        print("__init__", mp.current_process().name)
        self.name=None

    def run(self, name):
        self.name=name
        print("run", self.name)
        print("run", mp.current_process().name)

if __name__=='__main__':
    w=Worker()#main process에서 Worker가 생성되었기에 MainProcess로 출력된다. 다만 run()호출시 SubProcess로 출력된다.
    p=mp.Process(target=w.run, name='SubProcess', args=('bob',))
    p.start()
    p.join()

    print(w.name)#None(run에서 설정되는 self.name=name으로 인해 SubProcess가 출력될 것이라 생각할수 있지만, run은 subprocess에서 분리되어 실행되었기에 main_process에서 접근하지 않는다_상관X)

if __name__=="__main__":#위의 내용을 보여주는 또다른 예시상황
    w=Worker()
    p=mp.Process(target=w.run, name='SubProcess', args=('bob',))
    p.start()

    print('before join')
    print('subproces: ', p.is_alive())#True
    print(w.name)#None

    p.join()

    print("after join")
    print('subprocess: ', p.is_alive())#FALSE
    print(w.name)#None_main process에서 sub process의 값을 읽을 수 없다.

#타겟으로의 클래스 전달
    
class Worker:
    def __init__(self, name):
        print("__init__", mp.current_process().name)
        self.name=name
        self.run()

    def run(self):
        print('run', self.name)
        print('run', mp.current_process().name)

if __name__=='__main__':
    p=mp.Process(target=Worker, name='SubProcess', args=('bob',))#클래스 자체를 전달할 경우. 
    p.start()#시에 call. 마찬가지로 . init후 run을 호출한다. in subprocess
    p.join()
#__init__ SubProcess
#run bob
#run SubProcess
