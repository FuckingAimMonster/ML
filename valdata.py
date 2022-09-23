from xml.etree.ElementTree import parse
import os

path = 'C:/Users/yg058/Desktop/study/DeepLearning/capstone/valdata/Cab/body_annotation'
imgpath = 'C:/Users/yg058/Desktop/study/DeepLearning/capstone/valdata/Cab/img'
flist = os.listdir(path)

file = open("train.txt", "a", encoding='UTF-8')
class_id = 0
strs = ''
for fname in flist:
    xml_file = os.path.join(path, fname)
    tree = parse(xml_file)
    root = tree.getroot()
    fname = fname.rstrip('.xml') + '.jpg'
    strs = imgpath + '/' + fname
    for object in root.iter('object'):
        strs += ' '
        strs += object.find('bndbox').findtext('xmin') + ','
        strs += object.find('bndbox').findtext('ymin') + ','
        strs += object.find('bndbox').findtext('xmax') + ','
        strs += object.find('bndbox').findtext('ymax') + ','
        strs += '0'
    strs += '\n'
    file.write(strs)
    strs = ''
    
file.close()