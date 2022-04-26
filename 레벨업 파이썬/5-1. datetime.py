import datetime

now=datetime.datetime.now()#현재의 시간
print(now)

print(datetime.datetime(2022, 1, 15, 13, 51, 29, 865559))#지정하는 시간

today=datetime.datetime(
    year=now.year,
    month=now.month,
    day=now.day,
    hour=0,
    minute=0,
    second=0,
)#직접 인자에서 now데이터에 접근할 수 있다.

#timedelta: duration!
delta=datetime.timedelta(days=1)
print(delta, now+delta)
