
from eeg_method.dataset import DataSet
from eeg_method.define import *
from eeg_method.method.cca import CCA_Method
from eeg_method.method.trca import TRCA_Method
from eeg_method.method.method_paraloader import CCA_parameter_loader

#读取数据集
dataset=DataSet(file_address="G:\Course\MIND_LAB\SSVEP\lesson\dataset1")

#添加数据集的基本信息(必要，详见参数)
dataset.add_dataset_basicinfo(fs=250,info_tuple=(0,1,2,3))


#添加数据集的附加信息(非必要，详见参数)
otherinfo=Parameter_DataOtherInfo()
otherinfo.startcut=125
otherinfo.elected_channels=(48, 54, 55, 56, 57, 58, 61, 62, 63)
otherinfo.kfold_splits=1 #交叉验证分割数(CCA不分割)

#装载附加信息到数据集
dataset.add_dataset_otherinfo(otherinfo)

#读取根目录下的数据文件(.mat格式)
dataset.load_data(file_name="S3")

#添加滤波器参数(滤波器参数类，详见define.py,为Parameter_Filiter_XX类)
fliter_band=Parameter_Filiter_Band()
fliter_band.ws_wp=(6,70)    #通带频率范围
fliter_band.fliter_style=Define_Filiter.CHEBY  #滤波器类型
fliter_band.n=5#滤波器阶数

#执行滤波器
dataset.process_filter(fliter_band)

targets=(8,9,10,11,12,13,14,15,8.2,9.2,10.2,11.2,12.2,13.2,
        14.2,15.2,8.4,9.4,10.4,11.4,12.4,13.4,14.4,15.4,8.6,9.6,10.6,
        11.6,12.6,13.6,14.6,15.6,8.8,9.8,10.8,11.8,12.8,13.8,14.8,15.8)
cca_parameter_loader=CCA_parameter_loader(frq_tuple=targets,num_harmony=3,window_length=0.8)

#创建CCA方法实例(传入数据集实例和参数加载器实例)
cca_method=CCA_Method(dataset=dataset,cca_parameter_loader=cca_parameter_loader)


#训练/测试数据
acc,itr=cca_method.fit()
print(acc,itr)