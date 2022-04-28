import multiprocessing as mp
import time

def work():
    while True:#무한 loop에 유의!
        print("sub process is running")
        time.sleep(1)

if __name__=="__main__":
    p=mp.Process(target=work, name="SubProcess")#서브 프로세스의 생성 및 work실행(sleep1)
    print("Status: ", p.is_alive())#살아있는지 확인

    p.start()#서브프로세스 시작
    print("Status: ", p.is_alive())#살아있는지

    time.sleep(5)
    p.kill()#5초뒤 프로세스를 죽인다. 인 줄알았는데? 백그라운드에서 계속 실행된다고 한다.? 무튼 죽긴 죽었으니 work inf loop를 탈출했을텐데
    print("Status: ", p.is_alive())#살아있는지

    p.join()#실행과 동시에 기다린다_궁금한건 어케 끝나는건지 궁금. work는 무한루프인데 timeout이 default로 지정되어있나?

    print("Status: ", p.is_alive())#살이있는지
#예측: False, True, False, True / 결과: False, True, True(kill시 백그라운드에서 실행된다고 한다), False(마지막 join은 다 끝날 때 까지 기다리니 False)
#음.. 이거 굉장히 불친절하므로 타고난 내가 추측을 해보자면(어차피 제대로는 나가서 배울거니..)
#sub process 즉 main process와 분리되어있기에 p.kill()을 main process에서 호출시 즉각적으로 sub process가 kill되지 않아서 3번째 status에서 True가 떴고,
#main process에서 완전한 종료를 확인하고자 해서 join()으로 기다린 다음 확인을 했기에 False가 나왔다! 완벽할듯.   

#연습느낌으로 sub process에서 실행시키고자 하는 work가 클래스 일 경우 예시
import multiprocessing as mp
import time

class Work:
    def __init__(self):
        pass

    def run(self):
        while True:
            print("Sub process is runnig")
            time.sleep(1)

if __name__=='__main__':
    w=Work()
    p=mp.Process(target=w.run, name="SubProcess")
    print("Status: ", p.is_alive())#False

    p.start()
    print("Status: ", p.is_alive())#True 참고로 잠시 본질을 잊고 있었는데, 서브 프로세스의 상태는 is_alive()메서드 호출로 확인

    time.sleep(5)
    p.kill()#다시 본질. 프로세스 종료는 kill()메서드 호출.
    print("Status: ", p.is_alive())#True

    p.join()
    print("Status: ", p.is_alive())#False
