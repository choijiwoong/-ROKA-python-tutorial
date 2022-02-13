#[1. Memory]
print(id(2))#get address
print(hex(id(4)))

#activate garbage collection. but not recommanded
import gc

#optimize
import time

#first version
start_time=time.time()
a=list(range(100000))
a2=list()

for i in a:
    a2.append(i*2)#reallocation is occured!

end_time=time.time()
fin=end_time-start_time
print(fin)

#second version(comprehension)
start_time=time.time()
temp=[x*2 for x in range(100000)]
end_time=time.time()

fin=end_time-start_time
print(fin)

#thrid version(mapping)
start_time=time.time()
a=list(range(100000))
a2=map(lambda n: n*2, a)#map() manages python's operations & memory innerlly
end_time=time.time()

fin=end_time-start_time
print(fin)


#[2. manage file & directory]

#get working directory & change it
import os
print("current working directory is ", os.getcwd())
os.chdir("C:\\Users\\admin0!\\Desktop\\_2jimo")
print("changed working directory is ", os.getcwd())

#get list of file&directory in some directory
print("file&directory list in current directory: ",os.listdir())

#get list of file&directory that has same name
import glob
print("pdf contents: ", glob.glob('*.pdf'))

#check sub directories by using os.walk()
for root, dirs, files in os.walk(os.getcwd()):
    #print("all files&directories: ", files)
    print()

#if we want to specific files
for root, dir, files in os.walk(os.getcwd()):
    for f in files:
        a=os.path.splitext(f)[-1]#split ext
        if a=='.cpp':
            print(f)

#IsExist file or directory
print(os.path.exists("/Herry/poter"))
print(os.path.isdir("/C"))#check directory only
print(os.path.isfile("Herry/test.txt"))#check file only


#copy, move, delete of file&directory(feat. shutil)
import shutil
shutil.copy("data.xml", "data2.xml")#copy file; data.xml to data2.xml
shutil.copytree("asset", "data")#copy directory
shutil.move("data.xml", "data2.xml")#move file
shutil.rmtree('asset')#remove directory
os.remove("data.xml")#remove file

#get file size
print(os.path.getsize("test.txt"))
