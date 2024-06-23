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


def xml2txt(xmlPth, txtDir, copeImgDir, NoBoxes):
    """
    判断图像是否存在
    """
    # -----------------

    xmlFile = open(str(xmlPth), encoding='utf-8')
    xmlText = xmlFile.read()
    root = ET.fromstring(xmlText)
    pth = root.find('path').text
    fileName = root.find('filename').text
    txtFileName = fileName.replace("jpg", "txt")
    if not os.path.exists(pth):
        updir, _ = os.path.split(xmlPth)
        pth = os.path.join(updir, fileName)  # img path
    object_lst = list(root.iterfind('object'))
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    if len(object_lst) == 0:
        shutil.copy(pth, NoBoxes)
        xmlFileName = fileName.replace("jpg", "xml")
        updir, _ = os.path.split(xmlPth)
        xmlpth = os.path.join(updir, xmlFileName)
        shutil.copy(xmlpth, NoBoxes)
    else:
        txtPth = os.path.join(txtDir, txtFileName)
        with open(txtPth, 'w') as txtFile:
            for object in object_lst:
                label = object.find("name").text
                if label not in LABELS:  # 如果label不在预期中，不做处理。
                    continue
                else:
                    xMin = float(object.find("bndbox").find("xmin").text)
                    yMin = float(object.find("bndbox").find("ymin").text)
                    xMax = float(object.find("bndbox").find("xmax").text)
                    yMax = float(object.find("bndbox").find("ymax").text)
                    bb = convert((w, h), (xMin, xMax, yMin, yMax))
                    label = object.find("name").text
                    idx = LABELS[label]
                    txtFile.write(f"{idx} {' '.join([str(a) for a in bb])}\n")
            shutil.copy(pth, copeImgDir)
        xmlFile.close()
        # print(f"image label is {label}")
        return 1


def loop(subXmlLst, txtDir, copeImgDir, NoBoxes):
    print("\n")
    for xmlPth in tqdm.tqdm(subXmlLst):
        xml2txt(xmlPth, txtDir, copeImgDir, NoBoxes)


def mkdir(pth):
    if not os.path.exists(pth):
        os.makedirs(pth,exist_ok=True)


def xml_compare_img(xmlDir, copypth):
    # 剔除掉xml与jpg不匹配的文件，并没有删除，而是拷贝出来。
    img_lst = [os.path.basename(str(filename)).split('.')[0] for filename in list(pathlib.Path(xmlDir).glob('*.jpg'))]
    xml_lst = [os.path.basename(str(filename)).split('.')[0] for filename in list(pathlib.Path(xmlDir).glob('*.xml'))]

    if len(xml_lst) >= len(img_lst):
        t1 = img_lst
        t2 = xml_lst
    else:
        t1 = xml_lst
        t2 = img_lst
    for t in tqdm.tqdm(t1):
        if t in t2:
            shutil.copy(os.path.join(xmlDir, t + '.jpg'), copypth)
            shutil.copy(os.path.join(xmlDir, t + '.xml'), copypth)


if __name__ == '__main__':
    LABELS = {'TOUKUI':"0",
          'TOU':"1",
          "FXP":"2",
          "BS":"3"}
    xmlDir = r"E:\cope_data\Helmet_data\mtc-toukui"  # xml和img混合的原始数据
    copypth = r'E:\cope_data\Helmet_data\copydata'  # 对xmlDir处理后的数据
    NoBoxes = r'E:\cope_data\Helmet_data\NoBoxes'  # 无框的
    txtDir = r"E:\cope_data\Helmet_data\train\txt"  # xml 2 txt  save  txt
    copeImgDir = r"E:\cope_data\Helmet_data\train\img"  # xml 2 txt  save  img
    mkdir(copeImgDir)
    mkdir(txtDir)
    mkdir(NoBoxes)
    mkdir(copypth)
    print("处理xml与不匹配的图像...")
    xml_compare_img(xmlDir, copypth)

    xmlLst = list(pathlib.Path(copypth).glob("*.xml"))
    sumXml = len(xmlLst)
    dutyLst = []
    mpLst = []
    loop(xmlLst, txtDir, copeImgDir, NoBoxes,)
    
