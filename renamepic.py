import os
cnt = 365
path = '/home/mm/img_process/img_old/'
newpath = '/home/mm/img_process/img_new/'
img_total = os.listdir(path)

for i in img_total:
    print(i)
    oldname = path + str(i)
    newname = newpath + str(cnt) +".jpg"
    os.rename(oldname,newname)
    cnt = cnt +1
	

