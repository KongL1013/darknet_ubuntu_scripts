import os
import random
 
trainval_percent = 0.05
train_percent = 0.95

cc = os.getcwd()
xmlfilepath = cc+'/Annotations'
txtsavepath = cc+'/ImageSets/Main'
total_xml = os.listdir(xmlfilepath)
 
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
 
ftrainval = open('%s/ImageSets/Main/trainval.txt'%(cc), 'w')
ftest = open('%s/ImageSets/Main/test.txt'%(cc), 'w')
ftrain = open('%s/ImageSets/Main/train.txt'%(cc), 'w')
fval = open('%s/ImageSets/Main/val.txt'%(cc), 'w')
 
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)
 
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
