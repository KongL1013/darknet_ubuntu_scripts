# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import imgaug as ia


import imgaug.augmenters as iaa
import imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import numpy as np
import  os

newname_start = 2300
classes=["door","pipe","doc","table"]

path='/home/dd/img/pic_old/'
xml_path='/home/dd/img/xml_old/'


store_path_img='/home/dd/img/pic_new/'
store_path_xml='/home/dd/img/xml_new/'

ia.seed(1)

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

#the way to process
seq = iaa.Sequential([
    # Blur each image with varying strength using
    # gaussian blur (sigma between 0 and 3.0),
    # average/uniform blur (kernel size between 2x2 and 7x7)
    # median blur (kernel size between 3x3 and 11x11).
    iaa.OneOf([
        iaa.GaussianBlur((0, 5.0)),
        #iaa.AverageBlur(k=(2, 7)),
        iaa.MedianBlur(k=(3, 7)),
        # iaa.Multiply((0.85, 1.15), per_channel=0.5),
    ]),

# Change brightness of images (50-150% of original value).
#     iaa.Multiply((0.85, 1.15), per_channel=0.5),
#     iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
    iaa.AddToHueAndSaturation((-15, 15)),  # change their color
    # iaa.ContrastNormalization((0.9, 1.2)), # Strengthen or weaken the contrast in each image.
    # iaa.Affine(
    #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #     rotate=(-25, 25),
    #     shear=(-8, 8)
    # )

    # execute 0 to 5 of the following (less important) augmenters per image
    # don't execute all of them, as that would often be way too strong
    # iaa.SomeOf((0, 5),
    #            [
    #                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
    #                # convert images into their superpixel representation
    #                iaa.OneOf([
    #                    iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
    #                    iaa.AverageBlur(k=(2, 7)),  # blur image using local means with kernel sizes between 2 and 7
    #                    iaa.MedianBlur(k=(3, 11)),  # blur image using local medians with kernel sizes between 2 and 7
    #                ]),
    #                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
    #                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
    #                # search either for all edges or for directed edges,
    #                # blend the result with the original image using a blobby mask
    #                iaa.SimplexNoiseAlpha(iaa.OneOf([
    #                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
    #                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
    #                ])),
    #                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    #                # add gaussian noise to images
    #                iaa.OneOf([
    #                    iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
    #                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
    #                ]),
    #                iaa.Invert(0.05, per_channel=True),  # invert color channels
    #                iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
    #                iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
    #                # either change the brightness of the whole image (sometimes
    #                # per channel) or change the brightness of subareas
    #                iaa.OneOf([
    #                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
    #                    iaa.FrequencyNoiseAlpha(
    #                        exponent=(-4, 0),
    #                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
    #                        second=iaa.LinearContrast((0.5, 2.0))
    #                    )
    #                ]),
    #                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
    #                iaa.Grayscale(alpha=(0.0, 1.0)),
    #                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
    #                # move pixels locally around (with random strengths)
    #                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),  # sometimes move parts of the image around
    #                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
    #            ],
], random_order=True
)



def load_box(imge_name):
#more than one bbox
    name=imge_name.strip('.jpg')
    xml_name=name+'.xml'
    xml_localpath=os.path.join(xml_path, xml_name)
    #print(xml_localpath+"  "+imge_name)
    in_file=open(xml_localpath)
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    boxs=[]
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        #cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        onebox = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymax').text)]
        #print(str(cls_id)+" "+str(onebox))
        obj=[cls,onebox]
        boxs.append(obj)
    return boxs

def load_batch():
    # Images should be in RGB for colorspace augmentations.
    # (cv2.imread() returns BGR!)
    filelist = os.listdir(path)
    imgsread=np.array([], dtype=np.uint8)
    filelist.sort(key=lambda x:int(x[:-4]))  
    readlist=[]
    readbox=[]
    for f in filelist:
        dd = os.path.join(path, f)
        rgbimg = imageio.imread(dd)
        readlist.append(rgbimg)
        boxs=load_box(f)
        ibox=[dd,boxs]
        readbox.append(ibox)

    dread = np.asarray(readlist,dtype=np.uint8)
    #print(dread.shape)
    return dread,readbox



def make_xml(newname,h,w,c,bbs_aug):

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = str(newname) + '.jpg'

    # node_object_num = SubElement(node_root, 'object_num')
    # node_object_num.text = str(len(xmin_tuple))

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(w)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(h)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(c)

    for i in range(len(bbs_aug.bounding_boxes)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(bbs_aug.bounding_boxes[i].label)
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(bbs_aug.bounding_boxes[i].x1)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(bbs_aug.bounding_boxes[i].y1)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(bbs_aug.bounding_boxes[i].x2)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(bbs_aug.bounding_boxes[i].y2)


    xml = tostring(node_root, pretty_print = True)
    dom = parseString(xml)
    return dom


def wirte_img_xml(newname,img,bbs_aug):

    (R, G, B) = cv2.split(img)
    get = cv2.merge([B, G, R])
    img_path=os.path.join(store_path_img, str(newname) + '.jpg')
    cv2.imwrite(img_path,get)

    h,w,c=img.shape
    dom=make_xml(newname,h,w,c,bbs_aug)
    xml_name = os.path.join(store_path_xml, str(newname) + '.xml')
    with open(xml_name, 'wb') as f:
        f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))
if  __name__=='__main__':
    images,img_boxs_id=load_batch()
    for i in range(len(images)):
        current_image=images[i]
        all_box=[]
        for j in range(len(img_boxs_id[i][1])):
            current_box=BoundingBox(img_boxs_id[i][1][j][1][0], img_boxs_id[i][1][j][1][1], img_boxs_id[i][1][j][1][2], img_boxs_id[i][1][j][1][3],label=img_boxs_id[i][1][j][0])
            all_box.append(current_box)
        bbs = BoundingBoxesOnImage(all_box, shape=current_image.shape)
        image_aug, bbs_aug = seq(image=current_image, bounding_boxes=bbs)
        wirte_img_xml(newname_start, image_aug, bbs_aug)
        newname_start = newname_start + 1
