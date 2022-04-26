#./ run.py처럼 ./으로 파이썬 파일을 CMD에서 실행이 가능하다. 인자를 받을 경우엔 ./run.py -d i -f식으로 보내는데, 이러한 사용자 명령행의 인자를 파싱하는 것은 run.py의 몫이며, 이때 사용하는 모듈이 argparse이다.
#-d는 어떤 추가인자 값을 하나 받는 형태이고, -f는 더이상 추가인자가 필요없다는 형태이다.

#run.py에서 명령행을 파싱하기위해 argparse모듈을 임포트한 뒤, add_argument메서드로 파싱할 인자를 추가하면 된다.
#추가 옵션을 받을 경우 action='store', 추가 옵션이 없을 경우 action='store_true', 옵션값은 dest인자로 지정한 변수에 저장된다.

import argparse

parser=argparse.ArgumentParser()#Argument Parser를 생성
parser.add_argument("-d", "--decimal", dest="decimal", action="store")#받을 인자를 지정. -d혹은 --decimal로 들어온 인자를 decimal 멤버변수에 저장하며, 추가 옵션을 저장한다.
parser.add_argument("-f", "--fast", dest="fast", action="store_true")#동일하고 추가옵션을 저장하지 않고 그냥 옵션의 유무만 저장한다.(maybe for except?)
args=parser.parse_args()#argument를 parsing한다.

print(args.decimal)#전달된 인자값 확인
print(args.fast)
#./run.py -d 1 -f 시 1, True 출력.


#옵션에 따라 다르게 동작해야하는 코드
if args.decimal=='1':
    print("decimal is 1")
if args.fast:
    print("-f option is used")


#사용 예시
import argparse

parser=argparse.ArgumentParser()

parser.add_argument(dest='dst', action='store')
parser.add_argument('--qp', dest='qp', action='store')
parser.add_argument('--configure', dest='configure', action='store')
args=parser.parse_args()

print(args.dst)
print(args.qp)
print(args.configure)


#command line argument가 아닌 특정 파일에 저장된 argument를 직접 건네줄 경우, 리스트를 넘겨준다.
import argparse

parser=argparse.ArgumentParser()
parser.add_argument(dest="width", action='store')
parser.add_argument(dest='height', action='store')
parser.add_argument('--frames', dest='frames', action='store')
parser.add_argument('--qp', dest='qp', action='store')
parser.add_argument('--configure', dest='configure', action='store')

args=parser.parse_args(['64', '56', '--frames', '60', '--qp', '1', '--configure', 'AI'])#parse_args를 기본호출하면 cmd에서 받고, 인자로 리스트를 건넬 수 있다.
print(args.width, args.height, args.frames, args.qp, args.configure)
