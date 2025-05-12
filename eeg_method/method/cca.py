from typing import Tuple
import numpy as np
from scipy.sparse.linalg import eigs
from math import log2
from eeg_method.dataset import DataSet
from eeg_method.method.method_paraloader import CCA_parameter_loader
from sklearn.cross_decomposition import CCA


class CCA_Method():
    '''
    数据集类,用于利用CCA方法加载和处理 EEG 数据集。
    该类提供了加载数据、添加基本信息、添加其他信息和添加滤波器参数等功能。
    Args:
        dataset (DataSet): 数据集对象。
        window_length (float): 窗口长度，范围在(0,1]之间。
        frq_tuple (tuple): 目标刺激频率元组。
    Attention:
        CCA不划分训练测试集,而是将所有数据作为训练集,DataSet的KFold参数强制必须设置为1。
    
    Exception:
        ValueError: 如果frq_tuple的长度与目标数量不匹配。
        TypeError: 如果dataset不是DataSet的实例。
        TypeError: 如果cca_parameter_loader不是CCA_parameter_loader的实例。
        ValueError: 如果kfold_splits不等于1。
    '''
    def __init__(self,dataset:DataSet,cca_parameter_loader:CCA_parameter_loader) -> None:

        self.__init_parameter(dataset=dataset,cca_parameter_loader=cca_parameter_loader)
        self.train_data = self.dataset.train_data[:,0:self.time_points,:,:]
        self.train_labels=self.dataset.test_labels
        self.__parameter_check()
        self.__init_reference_signal()

        pass
    
    def __parameter_check(self) -> None:
        if not isinstance(self.dataset,DataSet):
            raise TypeError("dataset must be an instance of DataSet")
        if not isinstance(self.cca_parameter_loader,CCA_parameter_loader):
            raise TypeError("cca_parameter_loader must be an instance of CCA_parameter_loader")
        if len(self.frq_tuple)!=self.targets_num:
            raise ValueError("frq_tuple must be the same length as targets_num")
        if self.dataset.dataset_otherinfo.kfold_splits!=1:
            raise ValueError("kfold_splits must be 1 for CCA method")
  
    def __init_parameter(self,dataset:DataSet,cca_parameter_loader:CCA_parameter_loader) -> None:
        """
        初始化参数。
        """
        self.dataset = dataset
        self.cca_parameter_loader = cca_parameter_loader
        self.window_length=self.cca_parameter_loader.window_length
        self.elctored_channel,_,self.targets_num,self.trains_block=self.dataset.train_data.shape
        self.time_points=int(_*self.window_length)
        self.fs=self.dataset.fs
        
        self.frq_tuple = self.cca_parameter_loader.frq_tuple
        self.num_harmonics = self.cca_parameter_loader.num_harmony
        self.window_length = self.cca_parameter_loader.window_length
    
        self.itr_t=self.dataset.dataset_otherinfo.itr_t


    
    def __init_reference_signal(self) -> None:
        """
        初始化参考信号。
        """

        reference_signals = np.zeros(shape=(self.targets_num, self.num_harmonics * 2, self.time_points))
        t = np.arange(0, (self.time_points / self.fs), step=(1 / self.fs))
        for fre_idx in range(self.targets_num):
            for h in range(self.num_harmonics):
                reference_signals[fre_idx,h,:]=(np.sin(2 * np.pi * (h+1) * self.frq_tuple[fre_idx] * t)[0:self.time_points])
                reference_signals[fre_idx,h+1,:]=(np.cos(2 * np.pi * (h+1) * self.frq_tuple[fre_idx] * t)[0:self.time_points])
        self.reference_signals=reference_signals
        

    def __train_cca(self,x_np:np.ndarray,y_np:np.ndarray,components_num=1) -> np.ndarray:
        cca = CCA(components_num)#保留主成分量,default=1
        corr_np = np.zeros(components_num)
    
        result = np.zeros(self.targets_num)
        for fre_idx in range(0, self.targets_num):  #遍历所有频率
            cca.fit(x_np.T, y_np[fre_idx].T)
            x_a, y_b = cca.transform(x_np.T, y_np[fre_idx].T)
            for i in range(components_num):#遍历所有主成分
                corr_np [i] = np.corrcoef(x_a[:, i], y_b[:, i])[0, 1]
                result[fre_idx] = np.max(corr_np)
        return np.argmax(result)
           
    def fit(self) -> Tuple[float,float]:
        """
        执行CCA算法,输出ACC和ITR
        
        Args:
            None
        Returns:
            准确率:Acc,信息传输率:Itr
        """

        result = np.zeros((self.targets_num, self.trains_block))
        for train_idx in range(self.trains_block):
            for tar_idx in range(self.targets_num):
                matched_X = self.train_data[:, :, tar_idx,train_idx]
                matched_Y = self.reference_signals
                result[tar_idx,train_idx] = self.__train_cca(matched_X, matched_Y)
        
        is_correct = (result == self.train_labels)
        is_correct = np.array(is_correct).astype(int)
        test_acc = float(np.mean(is_correct))
        itr_val=CCA_Method.calculate_itr(n=self.targets_num, p=test_acc, t=self.itr_t)
    
        return test_acc,itr_val
        

    @classmethod  
    def calculate_itr(cls,n,p,t)->float:
        """
        计算信息传输率。

        Args:
            n (int): 类别数量。
            p (float): 准确率。
            t (float): 时间。

        Returns:
            float: 信息传输率。
        """
        if p < 0 or 1 < p:
            raise ValueError("p must in (0,1]")
        elif p < 1 / n:
            raise ValueError('The ITR might be incorrect because the accuracy < chance level.')
        elif p == 1:
            itr = log2(n) * 60 / t
        else:
            itr = (log2(n) + p * log2(p) + (1 - p) * log2((1 - p) / (n - 1))) * 60 / t
        return itr