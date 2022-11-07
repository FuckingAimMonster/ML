from xml.etree.ElementTree import parse
import os

path = 'C:/Users/yg058/Desktop/study/DeepLearning/capstone/valdata/Cab/testimg'
flist = os.listdir(path)

file = open("test.txt", "a", encoding='UTF-8')
class_id = 0
strs = ''
for fname in flist:
    jpg_file = path + '/' + fname
    strs += jpg_file
    strs += '\n'
    file.write(strs)
    strs = ''
    
file.close()