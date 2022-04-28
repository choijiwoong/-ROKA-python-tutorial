import multiprocessing as mp

#실행시킬 작업이 함수 버전일 경우의 예시
def work(value):
    pname=mp.current_process().name
    print(pname, value)

if __name__=="__main__":
    p=mp.Process(name='Sub Process', target=work, args=("hello",))#새 프로세스(Child)를 만든다. 
    p.start()
    p.join()
    print("Main Process")

#실행시킬 작업이 클래스 버전일 경우의 예시(args로 전한다는건 같다)
class Worker:
    def __init__(self):
        pass

    def run(self, value):#run으로만 잘 명시해주면 됨. 함수형으로 호출했을때의 인자가 run의 value로 마찬가지로 들어간다.
        pname=mp.current_process().name#python에서는 함수든 클래스든 다 객체여서 크게 다를 게 없다.
        print(pname, value)

if __name__=="__main__":
    w=Worker()
    p=mp.Process(name="Sub Process", target=w.run, args=("hello",))
    p.start()
    p.join()
    print("Main Process")
