import xml.etree.ElementTree as ET
import multiprocessing as mp
import pathlib
import os
import shutil
import tqdm




def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def xml2txt(xmlPth, imgDir,imagesDir, labelsDir):
    """
    xmlPth     xml路径
    imgDir     保存原始图像的路径
    imagesDir  保存新的img路径  like images/train
    labelsDir  保存新的txt路径  like labels/train
    判断图像是否存在
    """
    # -----------------
    # print(f"xmlPth: {xmlPth}")
    xmlFile = open(str(xmlPth), encoding='utf-8')
    xmlText = xmlFile.read()
    root = ET.fromstring(xmlText)
    bad_data = True
    if bad_data:
        fileName = os.path.basename(xmlPth)
        txtFileName = fileName.replace(".xml", ".txt")
    else:
        fileName = root.find('filename').text
        txtFileName = fileName.replace(".jpg", ".txt").replace(".png", ".txt")
    # try:
    #     pth = root.find('path').text
    # except:
    #     updir, _ = os.path.split(xmlPth)
    #     pth = os.path.join(updir, fileName)  # img path
    # updir, _ = os.path.split(xmlPth)
    if bad_data:
        imgpth = os.path.join(imgDir, fileName.replace('xml','jpg'))  # img path
    else:
        imgpth = os.path.join(imgDir, fileName)  # img path
    if not os.path.exists(imgpth):
        return -1
    shutil.copy(imgpth,imagesDir)
    object_lst = list(root.iterfind('object'))
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    txtPth = os.path.join(labelsDir,txtFileName)
    
    if len(object_lst) == 0:  # 无框的单独一个目录中去
        
        with open(txtPth,'w') as txtFile:
            txtFile.write('')
    else:
        with open(txtPth, 'w') as txtFile:
            for object in object_lst:
                label = object.find("name").text
                if (label in CLASSES) or (len(CLASSES)==0):  # 如果label不在预期中，不做处理。
                    xMin = float(object.find("bndbox").find("xmin").text)
                    yMin = float(object.find("bndbox").find("ymin").text)
                    xMax = float(object.find("bndbox").find("xmax").text)
                    yMax = float(object.find("bndbox").find("ymax").text)
                    bb = convert((w, h), (xMin, xMax, yMin, yMax))
                    label = object.find("name").text
                    idx = LABELS[label]
                    txtFile.write(f"{idx} {' '.join([str(a) for a in bb])}\n")
                else:
                    continue
                
    xmlFile.close()
    return 1


def loop(imgDir,xmlLst,imagesDir, labelsDir):
    print("\n")
    for xmlPth in tqdm.tqdm(xmlLst):
        xml2txt(xmlPth, imgDir,imagesDir, labelsDir)


def mkdir(pth):
    if not os.path.exists(pth):
        os.makedirs(pth,exist_ok=True)





if __name__ == '__main__':
   
    LABELS =  {'sfs_zc': 0}
    CLASSES = list(LABELS.keys())
    base = r"D:\data\TF\about_sfs\cl_dir_sunning"
    imgDir = base  # xml转txt，保存img路径
    xmlDir = base
    respth = r"D:\data\TF\about_sfs\cl_dir_sunning\cl_dir_sunning_sfs" # 对xmlDir处理后的数据
    labelsDir = os.path.join(respth,"labels","train")  # xml转txt，保存txt路径
    imagesDir = os.path.join(respth,"images","train")  # xml转txt，保存txt路径

    LABELS =  {"zxj_sfsks":1,"zxj_sfsds":1,"zxj_sfszc":0}

    CLASSES = list(LABELS.keys())

    base = r"D:\data\TF\about_sfs\xiangyang_yolov5-code_sfs\datasets\yolodata"
    imgDir = os.path.join(base,"images")  # xml转txt，保存img路径
    xmlDir = os.path.join(base,"xmls")
    respth = r"D:\data\TF\about_sfs\xiangyang_yolov5-code_sfs\datasets\xiangyang_sfs" # 对xmlDir处理后的数据
    labelsDir = os.path.join(respth,"labels","train")  # xml转txt，保存txt路径
    imagesDir = os.path.join(respth,"images","train")  # xml转txt，保存txt路径

    mkdir(labelsDir)
    mkdir(imagesDir)

    # 多线程处理
    xmlLst = list(pathlib.Path(xmlDir).glob("*.xml"))
    print("xml转txt...")
    loop(imgDir,xmlLst,imagesDir, labelsDir)
  
