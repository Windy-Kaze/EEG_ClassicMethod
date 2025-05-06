
from eeg_method.dataset import DataSet
from eeg_method.define import *
from eeg_method.method.trca import TRCA_Method

#读取数据集
dataset=DataSet(file_address="G:\Course\MIND_LAB\SSVEP\lesson\dataset1")
#添加数据集的基本信息(必要，详见参数)
dataset.add_dataset_basicinfo(fs=250,info_tuple=(0,1,2,3))

#添加数据集的附加信息(非必要，详见参数)
otherinfo=Parameter_DataOtherInfo()
otherinfo.startcut=125
otherinfo.elected_channels=(48, 54, 55, 56, 57, 58, 61, 62, 63)
#添加附加信息到数据集实例
dataset.add_dataset_otherinfo(otherinfo)
#读取根目录下的数据文件(.mat格式)
dataset.load_data(file_name="S3")

#添加滤波器参数(滤波器参数类，详见define.py,为Parameter_Filiter_XX类)
fliter_band=Parameter_Filiter_Band()
fliter_band.ws_wp=(6,70)    #通带频率范围
fliter_band.fliter_style=Define_Filiter.BUTTER  #滤波器类型
fliter_band.n=4 #滤波器阶数

#执行滤波器
dataset.process_filter(fliter_band)

#创建TRCA方法实例(传入数据集实例和窗长)
trca_method=TRCA_Method(dataset=dataset,window_length=0.8)
#训练/测试数据
acc,itr=trca_method.fit()
