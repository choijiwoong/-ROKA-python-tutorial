#[1. casting]
hex(12)                 #to 16

a=0b0010                #if we use it like decimal, it automatically will be converted
print(a)                #2

print(int('0xc', 16))   #convert to other notation string by int()

ord('가')               #char to unicode
chr(44032)              #unicode to char


#[2. show list of variables on global & local]_globals(), locals()
a=1
b=2
print(globals())        #show all global variables

a=[1,2,3]
def temp():
    global a            #we will use global variable; a
    a=[4,5,6]           #now, not local variable
    d=[7,]
    print(locals())     #print local variable with it's value
temp()                  


#[4. print Console]
aa=30
print("%s is integer"%aa)#print as format

obv=dict(name='peter', age=24, score=85)
#from pprint import *
#pprint.pprint(obv, width=20, indent=4)


#[5. module]
import math

print(math.ceil(-3.14))
math.floor(3.14)
math.trunc(-3.14)

print(round(4.5), round(3.5))#o.5 if even, floor else if odd, ceil.(사사오입 원칙)

#[6. built-in arthmetic functions] https://open.kakao.com/o/gp6GHMMc
