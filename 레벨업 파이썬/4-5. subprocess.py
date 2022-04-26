#check_output
import subprocess

output=subprocess.check_output('tasklist')
data=output.decode('cp949')
lines=data.splitlines()

for line in lines:
    print(line)


#프로그램 실행
cmd='notepad'
subprocess.run(cmd, shell=True)


#프로그램 출력값 캡쳐(다른 프로그램을 suprocess.run으로 실행하며 해당 프로그램이 화면에 출력하는 값을 파이썬 변수로 받아온다)
result=subprocess.run(cmd, capture_output=True, shell=True, encoding='utf-8')#capture_output=True를 사용
print(result.stdout)#반환값의 stdout

#실행 시간이 긴 프로그램 실행(기존의 run후 캡쳐는 cmd가 끝난 뒤 모든 출력을 한번에 받는데, 실시간으로 받을 수 있다.)
import subprocess

cmd='notepad'
process=subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, encoding='utf-8')
while True:
    output=process.stdout.readline()
    if output=='' and process.poll() is not None:
        break
    if output:
        print(output.strip())
