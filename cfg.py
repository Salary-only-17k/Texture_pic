# [配置参数]    前景权重值这只大点
#~ random params
## 贴图与原图贴图比重
gender_alpha_a:float = 0.5
gender_alpha_b:float = 0.8
## 旋转角度
gender_angle_a:int = -90
gender_angle_b:int =  90
## 中心位置偏移
gender_centerxy_a:float = 0.2 # 0.15
gender_centerxy_b:float = 0.3 # 0.35
## 贴图缩放尺寸
gender_resiz_a1:float = 1.
gender_resiz_b1:float = 1 # 2.0
gender_resiz_a2:float = 1# 0.8
gender_resiz_b2:float = 1# 1.2
gender_resiz_a3:float = 1# 0.5
gender_resiz_b3:float = 1# 1.0
## 最小框阈值
min_bbox:int = 8
#~ other params
## 最大随机次数
num:int = 1
## 线程池大小
worker:int = 4
## 随机种子
random_seed = -1 


