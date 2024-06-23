import shutil
from pathlib import Path
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import random
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import copy
import numpy as np

"""
使用说明：
    1. 贴边框出要贴图的区域。
    2. ps扣除贴图，保存为png有透明度格式。
    3. 运行代码    
"""


class document(Document):
    """重写xml的Document类中的函数"""

    def writexml(self, writer, indent="", addindent="", newl=""):
        for node in self.childNodes:
            node.writexml(writer, indent, addindent, newl)

    
class Texture(object):
    def __init__(self,backgroud_img_dir:str,backgroud_xml_dir:str ,mask_dir:str,out_texture_dir:str,out_xml_dir:str,out_detxml_dir:str,out_seg_dir:str ,relabels:dict):
        super(Texture,self).__init__()    
        self.backgroud_img_dir = backgroud_img_dir    # 背景图
        self.backgroud_xml_dir = backgroud_xml_dir    # 背景图的xml
        self.mask_dir = mask_dir                      # mask贴图
        
        self.out_texture_dir = out_texture_dir        # 输出贴图位置
        self.out_xml_dir = out_xml_dir                # 输出xml位置
        self.out_detxml_dir = out_detxml_dir          # 输出详细xml位置
        self.out_seg_dir = out_seg_dir                # 输出语义分割位置
        for d in [out_texture_dir,out_xml_dir,out_detxml_dir,out_seg_dir]:
            os.makedirs(d,exist_ok=True)
        self.relabels = relabels                      # 检测label
        self.erode = cfg.erode                        # 180~20

    def _conver2str(self,lst:list):
        return [str(p) for p in lst]   # 转化路径

    def _read_Alpha(self,patch_img_pth:str):  # 读取mask贴图
        patch_img_Alpha = cv2.imread(patch_img_pth, cv2.IMREAD_UNCHANGED)
        b, g, r, img_Alpha = cv2.split(patch_img_Alpha)
        patch_img = cv2.merge([b,g,r])
        return patch_img, img_Alpha

    def _gender_texture(self,backgroud_img:cv2.Mat,backgroud_xml_value:dict,mask_pths_lst:list,rnum:int):
        """
        backgroud_img           背景图
        backgroud_xml_value     背景图xml文件内容
        patch_pths_lst mask     贴图列表
        rnum                    随机贴图次数
        """
        res_imgs_dct = dict()
        scale = 1
        for rn in range(rnum):
            ele_dct=dict(texture=cv2.Mat,seg=cv2.Mat,xml=dict,det_xml=dict)
            big_xml ={
                "filename": backgroud_xml_value["filename"],
                "width": backgroud_xml_value["width"],
                "height": backgroud_xml_value["height"],
                "depth": backgroud_xml_value["depth"],
                "folder": "",
                "objects": []
            }
            mask_xml = {
                "filename": backgroud_xml_value["filename"],
                "width": backgroud_xml_value["width"],
                "height": backgroud_xml_value["height"],
                "depth": backgroud_xml_value["depth"],
                "folder": "",
                "objects": []  # xml_value["objects"].append([label,[xMin, xMax, yMin, yMax]]) 
            }
            tmpbackgroud_img = copy.deepcopy(backgroud_img)
            bg_dark = np.zeros_like(backgroud_img[...,0])
            for indx,bbox_info in enumerate(backgroud_xml_value['objects']):
                if bbox_info[0] in self.relabels:
                    # 读取mask图像
                    mask_pth = random.choice(mask_pths_lst)
                    mask_img,mask_img_Alpha = self._read_Alpha(mask_pth)
                    # 解析xml中需要贴图区域
                    xmin, xmax, ymin, ymax = bbox_info[1]
                    xmin,ymin,xmax,ymax =  int(xmin),int(ymin),int(xmax),int(ymax)
                    subw, subh = xmax-xmin, ymax-ymin
                    mask_h,mask_w = mask_img.shape[:-1]
                    # 生成随机数
                    rs_h,rs_w = self._gender_resize(subw, subh,mask_h,mask_w)
                    centerxy = self._gender_centerxy(subw, subh)
                    angle = self._gender_angle()
                    # 旋转，偏移，缩放
                    mask_img = cv2.resize(mask_img,(0,0),fx=rs_w,fy=rs_h)
                    mask_img_Alpha =cv2.resize(mask_img_Alpha,(0,0),fx=rs_w,fy=rs_h)
                    rotation_matrix = cv2.getRotationMatrix2D(centerxy, angle, scale)
                    rotated_sub_image = cv2.warpAffine(mask_img, rotation_matrix, (subw, subh))
                    rotated_sub_a = cv2.warpAffine(mask_img_Alpha, rotation_matrix, (subw, subh))
                    # 解析详细贴图位置
                    y_values = np.sum(rotated_sub_a,axis=1)
                    x_values = np.sum(rotated_sub_a,axis=0)
                    try:
                        rsxmin,rsxmax = np.nonzero(x_values)[0][0],np.nonzero(x_values)[0][-1]
                        rsymin,rsymax = np.nonzero(y_values)[0][0],np.nonzero(y_values)[0][-1]
                    except:
                        rsxmin,rsxmax = 0,0
                        rsymin,rsymax = 0,0 
                    if (rsxmin - rsxmax !=0) and (rsymin - rsymax !=0):  # xmin, xmax, ymin, ymax
                        mask_xml["objects"].append([self.relabels[bbox_info[0]],[str(rsxmin+xmin),str(rsxmax+xmin),str(rsymin+ymin),str(rsymax+ymin)]])
                        big_xml["objects"].append([self.relabels[bbox_info[0]],[str(xmin), str(xmax), str(ymin), str(ymax) ]])
                    else:
                        big_xml["objects"].append([bbox_info[0],[str(xmin), str(xmax), str(ymin), str(ymax)]])
                    # 生成seg的mask图
                    rotated_sub_a[rotated_sub_a>0]= cfg.seg_label
                    bg_dark[ymin:ymax, xmin:xmax] = rotated_sub_a 
                    # 把贴图中背影区域换乘bg的背景区域
                    sub_img_bg = tmpbackgroud_img[ymin:ymax, xmin:xmax]
                    for ih in range(subh):
                        for iw in range(subw):
                            if rotated_sub_a[ih,iw] ==0:
                                rotated_sub_image[ih,iw] = sub_img_bg[ih,iw]
                  
                    alpha = self._gender_alpha()
                    beta = (1-alpha) 
                    if beta<=0:
                        beta = 0.1
                    # 权重叠加
                    tmpbackgroud_img[ymin:ymax, xmin:xmax] = cv2.addWeighted(rotated_sub_image,alpha,tmpbackgroud_img[ymin:ymax, xmin:xmax],beta,1)
                else:
                    big_xml["objects"].append([bbox_info[0],[str(xmin), str(xmax), str(ymin), str(ymax)]])
            ele_dct=dict(texture=tmpbackgroud_img,seg_pic=bg_dark,big_xml=big_xml,sml_xml=mask_xml)
            
            res_imgs_dct[rn] = ele_dct
            
        return res_imgs_dct
    def _gender_alpha(self):
        return round(random.uniform(cfg.gender_alpha_a,cfg.gender_alpha_b),2)

    def _gender_angle(self):
        return random.randint(cfg.gender_angle_a,cfg.gender_angle_b)
    
    def _gender_centerxy(self,subw:int, subh:int):
        a = random.choice([1,-1])
        b = random.choice([1,-1])
        bias_w = round(random.uniform(cfg.gender_centerxy_a,cfg.gender_centerxy_b),2)*subw*a
        bias_h = round(random.uniform(cfg.gender_centerxy_a,cfg.gender_centerxy_b),2)*subh*b
        return [round(subw//2-bias_w),round(subh//2-bias_h)]
    
    def _gender_resize(self,subw:int, subh:int,patch_h:int,patch_w:int):
        rs_w,rs_h =0.0,0.0
        def inner(a,b):
            if a < b*.25:
                c = round(random.uniform(cfg.gender_resiz_a1,cfg.gender_resiz_b1),1)
            elif b*.25 <= a <= b*.5:
                c = round(random.uniform(cfg.gender_resiz_a2,cfg.gender_resiz_b2),1)
            else:
                c = round(random.uniform(cfg.gender_resiz_a3,cfg.gender_resiz_b3),1)
            return c
        rs_h = inner(subh,patch_h)
        rs_w = inner(subw,patch_w)
        return rs_h,rs_w

    def loop(self):
        backgroud_img_pths_lst = self._conver2str(list(Path(self.backgroud_img_dir).glob('*.jpg'))+list(Path(self.backgroud_img_dir).glob('*.png')))
        mask_pths_lst = self._conver2str(list(Path(self.mask_dir).glob('*.png')))
        if cfg.num==0:
            cfg.num = len(mask_pths_lst)
        with ThreadPoolExecutor(max_workers=cfg.worker) as exec:
            [exec.submit(self._core,backgroud_pth,mask_pths_lst) for backgroud_pth in backgroud_img_pths_lst[:10]]
        
    def _core(self,backgroud_img_pth,mask_pths_lst):
        print(f">>>  cope {backgroud_img_pth}...")
        rnum = random.randint(1,cfg.num)
        backgroud_img = cv2.imread(backgroud_img_pth)
        # ~~~~~~~~~
        xml_file = os.path.basename(backgroud_img_pth).replace(".jpg",".xml").replace(".png",".xml")
        backgroud_xml_value = self._read_xml(os.path.join(self.backgroud_xml_dir,xml_file))
        # ~~~~~~~~~
        res_imgs_dct = self._gender_texture(backgroud_img,backgroud_xml_value,mask_pths_lst,rnum)
        for indx,imgdata in list(res_imgs_dct.items()):
            self._saveinfo(backgroud_img_pth,indx,imgdata)

    def _modify_labels(self,xml_value:dict):
        new_xml_value = xml_value
        for i,obj_info in enumerate(xml_value["objects"]):
            if obj_info[0] in list(self.relabels.keys()):
                new_xml_value["objects"][i][0] = self.relabels[obj_info[0]]
        return new_xml_value

    def _saveinfo(self,backgroud_img_pth:str,indx:int,imgdata:any):
        filename = os.path.basename(backgroud_img_pth).split(".")[0]
        # 保存造img
        cv2.imwrite(os.path.join(self.out_texture_dir,f"{filename}_{indx}.jpg"),imgdata['texture'])
        # 保存seg
        cv2.imwrite(os.path.join(self.out_seg_dir,f"{filename}_{indx}.jpg"),imgdata['seg_pic'])
        # 保存bigxml
        self._write_xml(os.path.join(self.out_xml_dir,f"{filename}_{indx}.xml"),imgdata['big_xml'])
        # 保存smlxml
        self._write_xml(os.path.join(self.out_detxml_dir,f"{filename}_{indx}.xml"),imgdata['sml_xml'])

    def _read_xml(self,img_pth:str,folder='Unknown'):
        xml_pth = img_pth.replace(".jpg",".xml").replace(".png",".xml")
        is_test = False
        if is_test:
            img = cv2.imread(img_pth)
            if isinstance(img,cv2.Mat):
                return
            font = cv2.FONT_HERSHEY_SIMPLEX
        xml_value = {
                "filename": '',
                "width": '',
                "height": '',
                "depth": '',
                "folder": "",
                "objects": []
            }
        
        xmlFile = open(str(xml_pth), encoding='utf-8')
        xmlText = xmlFile.read()
        root = ET.fromstring(xmlText)

        xml_value["filename"] = root.find('filename').text

        size = root.find('size')
        xml_value['width'] = size.find('width').text
        xml_value['height'] = size.find('height').text
        if not size.find('depth'):
            xml_value['depth'] = '3'
        else:
            xml_value['depth'] = size.find('depth').text
        xml_value['folder']= folder
        object_lst = list(root.iterfind('object'))
        
        if len(object_lst) != 0:
            for obj in object_lst:
                label = obj.find("name").text
                xMin = obj.find("bndbox").find("xmin").text
                yMin = obj.find("bndbox").find("ymin").text
                xMax = obj.find("bndbox").find("xmax").text
                yMax = obj.find("bndbox").find("ymax").text
                xml_value["objects"].append([label,[xMin, xMax, yMin, yMax]])  # 修改这里
        return xml_value
    def _write_xml(self,save_xml_pth: str, xml_value: dict, database_name='img2_done'):
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

        with open(save_xml_pth, "w") as f:
            xmlBuild.writexml(f, newl='\n', addindent='\t')


def test_api(cfg):
    ## 在哪些标签内贴图
    relabels = dict(area='area_gai')
    cfg.seg_label = 255
    #~ data src
    ## 原始图像和xml位置
    backgroud_dir = 'test_data_1/bg'  # 包含了img和xml
    backgroud_img_dir = backgroud_dir
    backgroud_xml_dir = backgroud_dir
    ## 贴图位置
    mask_dir = 'test_data_1/mask_fenbi'          
    ## 输出造的图像和xml的位置
    out_base = 'test_data_1'
    
    out_texture_dir = os.path.join(out_base,'out','bg_coped') 
    out_xml_dir =  os.path.join(out_base,'out','xml_coped')
    out_detxml_dir =  os.path.join(out_base,'out','mask_xml')
    out_seg_dir =  os.path.join(out_base,'out','seg_pic') 
    # cfg.IS_RELABEL = True
    cfg.gender_alpha_a = 0.4
    cfg.erode = 40
    func = Texture(backgroud_img_dir,backgroud_xml_dir ,mask_dir,out_texture_dir,out_xml_dir,out_detxml_dir,out_seg_dir ,relabels)
    func.loop()
def test_api2(cfg):
    ## 在哪些标签内贴图
    relabels = dict(ding_area='ding_area_gai')
    cfg.seg_label = 255
    #~ data src
    ## 原始图像和xml位置
    backgroud_dir = 'test_data_2/bg'  # 包含了img和xml
    backgroud_img_dir = backgroud_dir
    backgroud_xml_dir = backgroud_dir
    ## 贴图位置
    mask_dir = 'test_data_2/mask'          
    ## 输出造的图像和xml的位置
    out_base = 'test_data_2'
    
    out_texture_dir = os.path.join(out_base,'out','bg_coped') 
    out_xml_dir =  os.path.join(out_base,'out','xml_coped')
    out_detxml_dir =  os.path.join(out_base,'out','mask_xml')
    out_seg_dir =  os.path.join(out_base,'out','seg_pic') 
    # cfg.IS_RELABEL = True
    cfg.gender_alpha_a = 0.4
    cfg.erode = 40
    cfg.gender_angle_a:int = 0
    cfg.gender_angle_b:int =  0
    cfg.gender_centerxy_a:float = 0.1 # 0.15
    cfg.gender_centerxy_b:float = 0.0 # 0.35
    cfg.gender_alpha_a:float = 0.7
    func = Texture(backgroud_img_dir,backgroud_xml_dir ,mask_dir,out_texture_dir,out_xml_dir,out_detxml_dir,out_seg_dir ,relabels)
    func.loop()

if __name__ == "__main__":
    import cfg as cfg
    if cfg.random_seed >0:
        random.seed(cfg.random_seed)
    test_api(cfg)
    # test_api2(cfg)

    
