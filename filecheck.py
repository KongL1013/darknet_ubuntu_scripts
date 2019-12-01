import os
imgpath = '/home/mm/img_process/1114afternoon/pic/'
xmlpath = '/home/mm/img_process/1114afternoon/xml/'

img_total = os.listdir(imgpath)
xml_total = os.listdir(xmlpath)

for i in img_total:
    imgname_num,jpg = i.split('.')
    xml = xmlpath+str(imgname_num)+'.xml'
    if not os.path.exists(xml):
	os.remove(imgpath+i)
	print('xml did not exists:',imgname_num)
	 
    
