#기본 프로세스 동작
import multiprocessing as mp
import time

def work():
    print("Sub Process start")
    time.sleep(5)
    print("Sub Process end")

if __name__=="__main__":
    print("Main Process start")
    proc=mp.Process(name='Sub Process', target=work)#sun process생성. daemon을 원할 시 daemon=True로 활성화가 가능. 메인 프로세스 종료시 서브 프로세스도 종료된다
    proc.start()
    proc.join()#join
    print("Main Process end")
