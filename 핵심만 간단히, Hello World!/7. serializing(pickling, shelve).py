#[1. pickling(serialize)_save variable as binary file]_자바와 같이 보안에 존나게 취약!
#saving data as binary file
import pickle

data1=["elsar", 24, 171]
data2=25

with open('test.pickle', 'wb') as f:
    pickle.dump(data1, f)#save variable
    pickle.dump(data2, f)

#loading data as binary file
import pickle

with open('test.pickle', 'rb') as f:
    data1=pickle.load(f)
    data2=pickle.load(f)
print(data1)
print(data2)


#[2. shelve(serialize)_save variable as file]
#saving data as datafile
import shelve

data1=["elsar", 24, 171]
data2=["peter", 31, 175]

with shelve.open('test.db') as f:
    f['obv1']=data1
    f['obv2']=data2

#loading data as datafile
import shelve
with shelve.open('test.db') as f:
    print(f['obv1'])
    print(f['obv2'])
