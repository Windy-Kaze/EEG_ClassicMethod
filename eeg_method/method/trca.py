from typing import Tuple
import numpy as np
from scipy.sparse.linalg import eigs
from math import log2

from eeg_method.dataset import DataSet

class TRCA_Method():
    '''
    用于利用TRCA方法加载和处理 EEG 数据集。
    该类提供了加载数据、添加基本信息、添加其他信息和添加滤波器参数等功能。
    Args:
        dataset (DataSet): 数据集对象。
        window_length (float): 窗口长度，范围在(0,1]之间。
       
    Exception:
        ValueError: 如果window_length不在(0,1]之间。
        TypeError: 如果dataset不是DataSet的实例。
    
    '''
    def __init__(self,dataset:DataSet,window_length=0.8) -> None:
        self.window_length = window_length
        self.dataset = dataset
        self.itr_t=self.dataset.dataset_otherinfo.itr_t
        self.elctored_channel,_,self.targets_num,self.trains_block=self.dataset.train_data.shape
        self.time_points=int(_*self.window_length)
        self.test_blocks=self.dataset.test_data.shape[3]
        
        self.__parameter_check()
        self.__process_dataset()
        pass
    
    def __process_dataset(self) -> None:
        """
        处理数据集,将数据集划分为训练集和测试集及其标签,将其保存到类中，无需手动访问。
        """      
        self.test_data = self.dataset.test_data[:,0:self.time_points,:,:]
        self.test_label_array=self.dataset.test_labels
        self.train_data = self.dataset.train_data[:,0:self.time_points,:,:]
        
        ...
    def __parameter_check(self) -> None:
        """
        检查参数的有效性。
        """
        if self.window_length<=0 or self.window_length>1:
            raise ValueError("window_length must be in (0,1]")
        if not isinstance(self.dataset,DataSet):
            raise TypeError("dataset must be an instance of DataSet")
    
    def __train_trca(self) -> np.ndarray:
        """
        训练 TRCA 方法,计算权重矩阵和模板矩阵
        
        Returns:
            np.ndarray: 权重矩阵，形状为 (Nc, 1)。 
        """
        w_array = np.zeros(shape=(self.targets_num, self.elctored_channel))
        trainavr_array = np.zeros(shape=(self.elctored_channel,self.time_points,self.targets_num))
      
        for targ_i in range(self.targets_num):
            traindata = self.train_data[:, :,targ_i,:]
            trainavr_array[:,:,targ_i] = np.mean(traindata, 2)#￥￥在其他位置处理合并训练集数据较好
            w_tmp = self.__trca_method(traindata=traindata)
            w_array[targ_i, :] = np.real(w_tmp[:, 0]) #获取大特征向量

        q_array = trainavr_array
        return  w_array, q_array
           
    def __test_trca(self, w_array:np.ndarray,q_array:np.ndarray) -> np.ndarray:
        """
        测试 TRCA 方法,计算测试数据的分类结果
        Args:
            w_array (np.ndarray): 权重矩阵，形状为 (Nc, 1)。
            q_array (np.ndarray): 模板矩阵，形状为 (Nc, 1)。
        Returns:
            np.ndarray: 测试数据的分类结果，形状为 (Nc, 1)。
        """
        

        result_array = np.zeros((self.targets_num, self.test_blocks))  

        """
        这部分代码计算测试数据与每个类别模板之间的相关性：

        对测试和训练数据应用空间滤波器
        计算滤波后信号间的相关系数
        存储结果到三维矩阵r[频带,类别,试验]       
            
        """
        for targ_i in range(self.targets_num):
            test_data_onetarg = self.test_data[:,:, targ_i, :]
            r_array = np.zeros((self.targets_num, self.test_blocks))
            for class_i in range(self.targets_num):     
                train_data_oneclass = q_array[:, :,class_i]
                w_oneclass = w_array[class_i, :]
                
                for trial_i in range(self.test_blocks):
                    testdata_w = np.matmul(test_data_onetarg [:, :, trial_i].T, w_oneclass)
                    traindata_w = np.matmul(train_data_oneclass [:, :].T, w_oneclass)
                    r_tmp = np.corrcoef(testdata_w.flatten(), traindata_w.flatten())
                    r_array[class_i, trial_i] = r_tmp[0, 1]

            ''' *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
  
            '''
            r_array_max = np.argmax(r_array, axis=0)
            result_array[targ_i, :] = r_array_max 

        return result_array
    
    def fit(self) -> Tuple[float,float]:
        """
        训练 TRCA 方法并计算测试准确率和信息传输率。
        该方法会自动调用训练和测试方法，并返回测试准确率和信息传输率。
        
        """

        w_array,q_array= self.__train_trca()

        estimated = self.__test_trca(w_array,q_array)
        # Evaluation
        is_correct = (estimated == self.test_label_array)
        is_correct = np.array(is_correct).astype(int)
        test_acc = float(np.mean(is_correct))
        itr_val=TRCA_Method.calculate_itr(n=self.targets_num, p=test_acc, t=self.itr_t)

        return test_acc ,itr_val

   
    def __trca_method(self,traindata:np.ndarray) -> np.ndarray:
        """
        TRCA方法的实现,计算特征值和特征向量。。
            求解方程S·v = λ·Q·v=>(Q^(-1)·S)v = λ·v
            --S(s_array)(Shape: num_chans,num_chans)是协方差矩阵
            --Q(q_array)(Shape: num_chans,num_chans)是正则化矩阵
            --v(v_arrary)(Shape: num_chans,1)是特征向量
            --λ(Shape: num_chans,1)是特征值
            这些特征向量表示最大化跨试验相关性的空间滤波器
        **尽管它是一个类方法,不建议用户直接调用它并私自传入数据,该方法会在self.train_trca被自动调用。
        Args:
            traindata (np.ndarray): 训练数据,shape=(elctored_channels,self.time_points,1,self.blocks)。
        Returns:
            np.ndarray: TRCA 方法计算得到的权重矩阵，形状为 (Nc, 1)。
        """
        ...
  
        s_array = np.zeros(shape=(self.elctored_channel, self.elctored_channel))     
        """
        计算协方差矩阵S:
            遍历所有不同的试验对(i,j)
            对每个试验数据进行中心化（减去均值）
            计算两个试验间的协方差(x1·x2^T和x2·x1^T)并累加, S=E(X-E(X))*E(Y-E(Y))
            这步操作的目标是量化不同试验间的相似性
        """
        for trial_i in range(self.trains_block - 1):
            x1 = traindata[:, :, trial_i]
            x1 = x1 - np.expand_dims(np.mean(x1, 1), 1).repeat(x1.shape[1], 1)
            for trial_j in range(trial_i + 1, self.trains_block):
                x2 = traindata[:, :, trial_j]
                x2 = x2 - np.expand_dims(np.mean(x2, 1), 1).repeat(x2.shape[1], 1)
                s_array = s_array + np.matmul(x1, x2.T) + np.matmul(x2, x1.T)
        
        """
        计算正则化矩阵Q:
            对每个试验数据进行中心化（减去均值）
            计算正则化矩阵Q=E(X-E(X))*E(X-E(X))^T
            这步操作的目标是量化同一试验间的相似性
        """
        q_corr_array = traindata.reshape(self.elctored_channel, self.time_points * self.trains_block)
        q_corr_array  = q_corr_array  - np.expand_dims(np.mean(q_corr_array , 1), 1).repeat(q_corr_array .shape[1], 1)
        q_array = np.matmul(q_corr_array , q_corr_array .T)
        
        _, v_array= eigs(s_array,6,q_array) # type: ignore
        return v_array
          
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


    def __str__(self) -> str:
        ...
