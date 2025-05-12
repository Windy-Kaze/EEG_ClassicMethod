from typing import  Union,Any,Tuple
from sklearn.model_selection import KFold
import scipy.io
from scipy import signal
import numpy as np

from eeg_method.define import Define_Filiter, Parameter_DataOtherInfo, ParameterFilter
from eeg_method.display import wranning

class DataSet:
    '''
    数据集类，用于加载和处理 EEG 数据集。
    
    Args:
        file_address (str): 数据集文件地址。
        
    Exception:
        ValueError: 如果数据集的基本信息未设置。
        TypeError: 如果info不是Parameter_DataOtherInfo的实例。
        TypeError: 如果filter_paraclass不是ParameterFilter的实例。
。
    '''
    def __init__(self, file_address: str) -> None:   
        self.data_address = file_address
        self.fs=0
        self.info_indextuple=()
        
    def __class_hint(self):
        self.dataset_otherinfo:Parameter_DataOtherInfo

    def add_dataset_basicinfo(self,fs:Union[int,float],info_tuple:tuple)->None:
        """
        向数据集添加基本信息。

        Args:
            "fs" (int||float): 采样频率，例如 500。
            "eeginfo_tuple" (tuple): 包含 EEG 通道信息的元组,为(电极索引,数据点,目标索引,区块索引)的顺序索引:
                ->例1:数据集格式为(电极索引,数据点,目标索引,区块索引),即(0,1,2,3)
                ->例2:数据集格式为(目标索引,数据点,电极索引,区块索引),即(2,1,0,3)                   
        Returns:
            None: 此方法返回None。
        """
        self.fs = fs
        self.info_indextuple = info_tuple

    def add_dataset_otherinfo(self, info:Parameter_DataOtherInfo)->None:
        """
        向数据集添加其他信息。

        Args:
            info (Parameter_DataOtherInfo): 其他信息的参数类实例。
        Returns:
            None: 此方法返回None。
        """
        if not isinstance(info, Parameter_DataOtherInfo):
            raise TypeError("info must be an instance of Parameter_DataOtherInfo")
        
        self.dataset_otherinfo = info

    def load_data(self,file_name:str) -> None:
        """
        加载数据集并划分为训练集和测试集及其标签,将其保存到类中，无需手动访问。
        Args:
            file_name (str): 数据集文件名字(排除序号):
                ->例1:"S1.mat",则传入"S1"
        Returns:
            None:None
        """       
        def check_parameter_loading(self=self) -> None:
            if not hasattr(self, 'check_oncetime_flag'):
                self.__setattr__('check_oncetime_flag', True)
                if not self.fs:
                    raise ValueError("fs is not set")
                if not self.info_indextuple:
                    raise ValueError("info_indextuple is not set")        
        def split_data(data: np.ndarray, label: np.ndarray,self=self) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
            
            if(self.dataset_otherinfo.kfold_splits==1):
                train_data = data
                test_data = data
                train_labels = label
                test_labels = label
                return train_data, test_data, train_labels, test_labels
    
            kfold = KFold(n_splits=self.dataset_otherinfo.kfold_splits,shuffle=True, random_state=42)
            # 对最后一个维度（block）进行划分
            for train_idx, test_idx in kfold.split(range(data.shape[-1])):
                train_data = data[:, :, :, train_idx]  # 训练集
                test_data = data[:, :, :, test_idx]   # 测试集
                train_labels = label[:,train_idx]      # 训练集标签
                test_labels = label[:,test_idx]        # 测试集标签      
            return train_data, test_data, train_labels, test_labels
        
        check_parameter_loading()
        #获取mat的第一个数据集
        subjectfile = scipy.io.loadmat(file_name=f'{self.data_address}/{file_name}')
        data = subjectfile[list(subjectfile.keys())[-1]]
        data=data.transpose(self.info_indextuple) 
        
        start_cut,end_cut=self.dataset_otherinfo.startcut,self.dataset_otherinfo.endcut
        elected_channels=self.dataset_otherinfo.elected_channels

        if not hasattr(self, 'dataset_otherinfo'):
            wranning(msg="注意!!你并没有传入电极索引等其他信息\n\
                对于一些任务或未完全处理的数据集,这些信息是必须的！")
    
        data=data[elected_channels,start_cut:(np.size(data,1)-end_cut),:,:]
        self.elected_channels_num,self.time_points,self.target_frqnum,self.blocks=data.shape
    
        label_data = np.zeros((self.target_frqnum, self.blocks))
        for i in range(self.target_frqnum):
            label_data[i,:]=i
            
        self.train_data,self.test_data,self.train_labels,self.test_labels=split_data(data,label_data)
       
        
        
    def process_filter(self,filter_paraclass:ParameterFilter):
        """
        处理滤波器参数。
        Args:
            self.filter_paraclass (ParameterFilter): 滤波器参数抽象类的子类,不要直接传入它本身!。
        Returns:
            None: 此方法返回None。
        """

        if not hasattr(self, 'train_data'):
            raise ValueError("请执行load_data方法加载数据集")
        
        if not isinstance(filter_paraclass, ParameterFilter) or type(filter_paraclass) is ParameterFilter:
            raise TypeError("self.filter_paraclass must be an instance of a subclass of ParameterFilter") 
        
        self.filter=_Filter(parameterfilter=filter_paraclass,fs=self.fs)
        self.train_data=signal.filtfilt(self.filter.b, self.filter.a, self.train_data,axis=1)
        self.test_data=signal.filtfilt(self.filter.b, self.filter.a, self.test_data,axis=1)

            
    def __str__(self) -> str:
        ...


class _Filter:
    def __init__(self,parameterfilter:ParameterFilter,fs:float) -> None:
        self.filter_paraclass=parameterfilter
        self.fs=fs
        self.nyquist = 0.5 * self.fs 
        self.__parameter_init()
        self.__parameter_loader()
        self.__structure_filter(type=self.type,style=self.style)    
    
    def __parameter_init(self)->None:
        self.type=None
    def __parameter_loader(self)->None:
        self.type=self.filter_paraclass.type
        self.style=self.filter_paraclass.fliter_style
        self.n=self.filter_paraclass.n

    def __cheby1(self,n:int,Wn:Union[float,tuple],btype:str,rp=0.5):
        return signal.cheby1(N=n, rp=rp, Wn=Wn,btype=btype)
    

    def __structure_filter(self,type,style):
        if(style==Define_Filiter.BUTTER):
            my_filter=signal.butter
        elif(style==Define_Filiter.CHEBY):
            my_filter=self.__cheby1
 
        elif(style==Define_Filiter.NOTCH):
            my_filter=signal.iirnotch
    
        if(type==Define_Filiter.BAND):
            ws_wp=(self.filter_paraclass.ws_wp[0],self.filter_paraclass.ws_wp[1])
            low = ws_wp[0]/ self.nyquist
            high = ws_wp[1] / self.nyquist
            self.b, self.a = my_filter(self.n,(low, high), btype='band')

        elif(type==Define_Filiter.HIGH):
            self.ws=self.filter_paraclass.ws
            ws = self.ws / self.nyquist
            self.b, self.a = my_filter(self.n, ws, btype='high')
            
        elif(type==Define_Filiter.LOW):
            self.ws=self.filter_paraclass.ws
            ws = self.ws / self.nyquist
            self.b, self.a = my_filter(self.n, ws, btype='low')
            
        elif(type==Define_Filiter.NOTCH):
            self.notch_wp=(self.filter_paraclass.notch_wp)
            notch_wp = self.notch_wp / self.nyquist
            self.b, self.a = my_filter(notch_wp ,30)
    
    def process_filter(self):
        ...

    