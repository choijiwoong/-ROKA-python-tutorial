"""이터러블
a=1
print(iter(a))#iterable한지 확인하는 방법 TypeError: 'int' object is not iterable. 이는 for문을 사용할 수 없음을 의미한다.


a=1
for i in a:
    print(i)#iterable하지 않다 == for문을 사용할 수 없다.
"""

a=[1,2,3]
print(iter(a))#TypeError가 발생하지 않는다! iterable하다.

for i in a:
    print(i)#well-done!

#이터러블 객체(__iter__메서드 포함)와 이터레이터 객체(__iter__가 반환하는 객체로, __next__메소드가 구현되어있다.)
class MyIterator:
    def __next__(self):#next접근 시 1을 반환
        return 1

class MyIterable:
    def __iter__(self):#Iterable객체
        obj=MyIterator()#Iterator 객체 생성 및 반환
        return obj

m=MyIterable()
r=iter(m)
print(next(r))
print(next(r))
print(next(r),'\n\n')

#예시_ __iter__과 __next__메소드를 구현하면 된다. 
class Season:
    def __init__(self):
        self.data=['봄', '여름', '가을', '겨울']
        self.index=0

    def __iter__(self):#__next__가 구현되어있는 iterator객체를 리턴해야한다.
        return self#자기혼자 iterable, iterator다할거임

    def __next__(self):
        if self.index<len(self.data):
            cur_season=self.data[self.index]
            self.index+=1
            return cur_season

        else:
            raise StopIteration#iteration stop!
s=Season()
for i in s:
    print(i)
