import os
import tqdm
import shutil
import pathlib
from datetime import datetime as dt
import xml.dom.minidom as xdc
from xml.dom.minidom import Document
from xml.etree import ElementTree as ET
from xml.dom.minidom import parse
import cv2
class document(Document):
    """重写xml的Document类中的函数"""
    def writexml(self, writer, indent="", addindent="", newl=""):
        for node in self.childNodes:
            node.writexml(writer, indent, addindent, newl)


def xywh2xyxy(size,xywh):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = xywh[0] / dw
    y = xywh[1] / dh
    w = xywh[2] / dw
    h = xywh[3] / dh
    # print([x, y, w, h])
    x_min = round((2*x-w+2)*0.5)
    x_max = round((2*x+w+2)*0.5)
    y_min = round((2*y-h+2)*0.5)
    y_max = round((2*y+h+2)*0.5)
    return [x_min,x_max,y_min,y_max]


def read_txt(txtPth,img_dir):
    """
    xmlDir:str   保存生成xml的目录
    xml_value: dict
    """
    xml_value = {
            "filename": '',
            "width": '',
            "height": '',
            "depth": '',
            "folder": "",
            "objects": []
        }
    xml_value["filename"] = os.path.basename(txtPth).replace('.txt','.jpg')
    try:
        img = cv2.imread(os.path.join(img_dir,xml_value["filename"]))
        h,w,_ = img.shape
    except:
        return xml_value
    xml_value["width"] = str(w)
    xml_value["height"] = str(h)
    xml_value["depth"] = str(3)
    with open(str(txtPth),'r', encoding='utf-8') as txtFile:
        Textlines = txtFile.readlines()
        for line in Textlines:
            label_ind,x_r,y_r,w_r,h_r = line.split(' ')[0],line.split(' ')[1],line.split(' ')[2],line.split(' ')[3],line.split(' ')[4]
            label_ind,x_r,y_r,w_r,h_r = int(label_ind),float(x_r),float(y_r),float(w_r),float(h_r)
            xMin,xMax,yMin,yMax = xywh2xyxy([w,h],[x_r,y_r,w_r,h_r])
            try:
                xml_value["objects"].append([str(LABELs[label_ind]),[str(xMin), str(xMax), str(yMin), str(yMax)]])  # 修改这里
            except:
                print("label_ind >>>   ",label_ind,type(label_ind))
    return xml_value


def write_xml(xmlDir: str, xml_value: dict, database_name='img2_done'):
    """
    xmlDir:str   保存生成xml的目录
    xml_value: dict
    """
    xmlBuild = document()

    annotation = xmlBuild.createElement("annotation")
    xmlBuild.appendChild(annotation)

    folder = xmlBuild.createElement("folder")
    foldercontent = xmlBuild.createTextNode(xml_value["folder"])
    folder.appendChild(foldercontent)
    annotation.appendChild(folder)  # folder标签结束

    filename = xmlBuild.createElement("filename")
    filenamecontent = xmlBuild.createTextNode(xml_value["filename"])
    filename.appendChild(filenamecontent)
    annotation.appendChild(filename)  # folder标签结束

    path = xmlBuild.createElement("path")
    pathcontent = xmlBuild.createTextNode("Unknown")
    path.appendChild(pathcontent)
    annotation.appendChild(path)  # folder标签结束

    source = xmlBuild.createElement("source")
    annotation.appendChild(source)
    database = xmlBuild.createElement("database")
    databasecontent = xmlBuild.createTextNode(database_name)
    database.appendChild(databasecontent)
    source.appendChild(database)  # folder标签结束

    size = xmlBuild.createElement("size")
    annotation.appendChild(size)
    
    width = xmlBuild.createElement("width")
    widthcontent = xmlBuild.createTextNode(xml_value["width"])
    width.appendChild(widthcontent)
    size.appendChild(width)
    height = xmlBuild.createElement("height")
    heightcontent = xmlBuild.createTextNode(xml_value["height"])
    height.appendChild(heightcontent)
    size.appendChild(height)
    depth = xmlBuild.createElement("depth")
    depthcontent = xmlBuild.createTextNode(xml_value["depth"])
    depth.appendChild(depthcontent)
    size.appendChild(depth)

    segmented = xmlBuild.createElement("segmented")
    segmentedcontent = xmlBuild.createTextNode("0")
    segmented.appendChild(segmentedcontent)
    annotation.appendChild(segmented)  # folder标签结束
    if len(xml_value["objects"]) != 0:
        for obj_info in xml_value["objects"]:
            object = xmlBuild.createElement("object")
            annotation.appendChild(object)
            name = xmlBuild.createElement("name")
            namecontent = xmlBuild.createTextNode(obj_info[0])
            name.appendChild(namecontent)
            object.appendChild(name)
            
            pose = xmlBuild.createElement("pose")
            posecontent = xmlBuild.createTextNode("Unspecified")
            pose.appendChild(posecontent)
            object.appendChild(pose)
            truncated = xmlBuild.createElement("truncated")
            truncatedcontent = xmlBuild.createTextNode("0")
            truncated.appendChild(truncatedcontent)
            object.appendChild(truncated)
            difficult = xmlBuild.createElement("difficult")
            difficultcontent = xmlBuild.createTextNode("0")
            difficult.appendChild(difficultcontent)
            object.appendChild(difficult)
            bndbox = xmlBuild.createElement("bndbox")
            object.appendChild(bndbox)
            xmin = xmlBuild.createElement("xmin")
            xmincontent = xmlBuild.createTextNode(obj_info[1][0])
            xmin.appendChild(xmincontent)
            bndbox.appendChild(xmin)
            ymin = xmlBuild.createElement("ymin")
            ymincontent = xmlBuild.createTextNode(obj_info[1][2])
            ymin.appendChild(ymincontent)
            bndbox.appendChild(ymin)
            xmax = xmlBuild.createElement("xmax")
            xmaxcontent = xmlBuild.createTextNode(obj_info[1][1])
            xmax.appendChild(xmaxcontent)
            bndbox.appendChild(xmax)
            ymax = xmlBuild.createElement("ymax")
            ymaxcontent = xmlBuild.createTextNode(obj_info[1][3])
            ymax.appendChild(ymaxcontent)
            bndbox.appendChild(ymax)

    with open(os.path.join(xmlDir, xml_value["filename"].replace('.jpg', '.xml')), "w") as f:
        xmlBuild.writexml(f, newl='\n', addindent='\t')



class cope():
    def __init__(self,txt_dir,xml_dir,img_dir) -> None:
        super(cope,self).__init__()
        self.txt_dir = txt_dir
        self.xml_dir = xml_dir
        self.img_dir = img_dir
    def loop(self):
        txt_pths_lst = list(pathlib.Path(self.txt_dir).glob("*.txt"))
        for txt_pth in tqdm.tqdm(txt_pths_lst):
            txt_pth = str(txt_pth)
            if "classes.txt" in txt_pth:
                continue
            xml_value = read_txt(txt_pth,self.img_dir)
            write_xml(self.xml_dir,xml_value)
        

if __name__ == "__main__":

    LABELs = {0: 'cat',
                1: 'dog',
                2: 'mouse',
                3: 'face',}
    base = r"path/to/basedir"
    txt_dir = os.path.join(base,'labels')
    xml_dir = os.path.join(base,'xmls')
    img_dir = base
    os.makedirs(xml_dir,exist_ok=True)
    func = cope(txt_dir,xml_dir,img_dir)
    func.loop()