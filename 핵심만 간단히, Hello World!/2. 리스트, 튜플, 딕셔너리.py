#[1. list]

a=[1,2]
b=['apple', 'banana']

c=[1, 2, 'apple', 'banana']
d=[1, 2, ['apple', 'banana']]

e=[1, 2, ('apple', 'banana')]#() is const tuple.
f=[]                        #empty

print(a[0])                 #1
print(d[2][1])              #banana
print(e[1:3])               #[2, ('apple', 'banana')]_slicing

#comprehension
del a
a=[x*2 for x in range(6)]   #Comprehension way for good handling of list
print(a)                    #[0, 2, 4, 6, 8, 10]

del a
a=['apple', 'banana', 'orange']
print([x for x in a  if 'na' in x])#example of comprehension 2

#zip
del a,b,d
a=[1,2,3]
b=[4,5,6]
d=zip(a,b)
print(list(d))              #[(1,4), (2,5), (3,6)]

print()
#[2. tuple]
a=(1,2)
#del a[0]                    #error occur_TypeError: 'tuple' object doesn't support item deletion
#a[0]=1                      #error occur_TypeError: 'tuple' object does not support item assignment

a=1                         #also tuple
a=('a', 'b', ('c', 'd'))
a=(1,2,[3,4])
a[2][0]=6                   #modifiable

a=()                        #empty
a=(1,)                      #If we want to add one element, we must use ','

a=(1,2,'c')
print(a[1])                 #2
print(a[1:3])               #(2, 'c')

a=('a','b',('c','d'))
print(a[1])                 #b
print(a[2][0])              #c

print()
#[3. dictionary]
del(a)
a={'name': 'hong', 'age': 24}
print(a)                    #{'name': 'hong', 'age': 24}
print(a['name'])            #'hong'

a['height']=189             #add element
del a['height']             #del
print(a.keys())             #get_key method
print(a.values())           #get_value method
print(a.items())            #get_element method
a.clear()
print(a)                    #{}

#Make Dictionary by using Comprehension
a=['apple', 'banana', 'orange']
b=[7,5,8]
d={x:y for x, y in zip(a,b)}#{'apple': 7, 'banana': 5, 'orange': 8}
print(d)
