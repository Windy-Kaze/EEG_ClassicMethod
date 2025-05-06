
class Define_Dataset:
    KFOLD_SPLITER_DEFAULT=5

class Flag_counter:
    __count=-1
    @classmethod
    def count(cls):
        cls.__count+=1
        return cls.__count

class Define_Filiter:
    _LOW=Flag_counter.count()
    _HIGH=Flag_counter.count()
    _BAND=Flag_counter.count()
    _NOTCH=Flag_counter.count()
    _MAX_CHEBY_N=5
    
    BUTTER=Flag_counter.count()
    CHEBY=Flag_counter.count()

    FILITER_STYLE=(BUTTER,CHEBY)


class ParameterFilter():
    """
    抽象类，表示滤波器的参数配置。

    Attributes:
        _type (int): 滤波器类型，由子类定义。
        fliter_style (int): 滤波器样式，默认为 -1。
    """

    fliter_style: int = -1
    ws_wp = (0,0)
    ws=0.0
    
    @property
    def type(self):
        """
        获取滤波器类型（只读）。
        """
        pass

class Parameter_DataOtherInfo():
    """
    数据集的其他信息配置类。
    Attributes:
        elctrode (tuple): 选定的电极通道，默认为空元组。
        startcut (int): 数据切割的起始索引,默认为0。
        endcut (int): 数据切割的结束索引,默认为0。
        kfold_splits (int): KFold 划分的折数，默认为 5。
        itr_t(float||int): 用于计算信息传输率,每次选择所用的时间,默认为0.5。
    
    """
    
    _elected_channels = slice(None)
    _startcut,_endcut=0,0
    _train_num=0
    _kfold_splits=Define_Dataset.KFOLD_SPLITER_DEFAULT
    _itr_t=0.5

    @property
    def itr_t(self):
        return self._itr_t
    @property
    def kfold_splits(self):
        return self._kfold_splits
    
    @property
    def elected_channels(self):
        return self._elected_channels
    @property
    def startcut(self):
        return self._startcut

    @property
    def endcut(self):
        return self._endcut 
    @property
    def train_num(self):
        return self._train_num
    
    @itr_t.setter
    def itr_t(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("The attribute must be of type int or float.")
        self._itr_t = value
        
    @kfold_splits.setter
    def kfold_splits(self, value):
        if not isinstance(value, int):
            raise TypeError("The attribute must be of type int.")
        self._kfold_splits = value
        
    @train_num.setter
    def train_num(self, value):
        if not isinstance(value, int):
            raise TypeError("The attribute must be of type int.")
        self._train_num = value
    
    @startcut.setter
    def startcut(self, value):
        if not isinstance(value, int):
            raise TypeError("The attribute must be of type int.")
        self._startcut = value
        
    @endcut.setter
    def endcut(self, value):
        if not isinstance(value, int):
            raise TypeError("The attribute must be of type int.")
        self._endcut = value
        
    @elected_channels.setter
    def elected_channels(self, value):
        if not isinstance(value,(list,tuple)):
            raise TypeError("The attribute must be of type tuple or list.")
        self._elected_channels = value   


class Parameter_Filiter_Low(ParameterFilter):
    """
    表示低通滤波器的参数配置。

    Attributes:
        type (int): 滤波器类型，固定为 Define_Filiter.LOW。
        fliter_style (str): 滤波器样式，默认为 -1。
        ws (float): 滤波器的截止频率，必须为 int 或 float 类型。

    Methods:
        type: 获取滤波器类型（只读）。
        ws: 获取或设置滤波器的截止频率。
            - 设置时，必须为 int 或 float 类型，否则抛出 TypeError。
    """
    def __init__(self):
        self._fliter_style:str=""
        self._ws=0
        self._n=4
    

    def check_n(self):
        if self._fliter_style=="":
            raise ValueError("The 'fliter_style' attribute must be set before checking 'n'.")
        
        if self._fliter_style==Define_Filiter.CHEBY:
            if self._n>Define_Filiter._MAX_CHEBY_N:
                raise ValueError("The 'n' attribute must be less than or equal to Define_Filiter._MAX_CHEBY_N.")
            
    @property
    def n(self):
        return self._n

    @property
    def type(self):
        return self._type
    
    @property
    def ws(self):
        return self._ws
    
    @property
    def fliter_style(self):
        return self._fliter_style
    
    @n.setter
    def n(self, value):
        if not isinstance(value, int):
            raise TypeError("The 'n' attribute must be of type int.")
        self.check_n()
        self._n = value
        
    @fliter_style.setter
    def fliter_style(self, value):
        if value not in Define_Filiter.FILITER_STYLE:
            raise ValueError("The 'fliter_style' attribute must be one of the defined filter styles.Given by (BUTTER,CHEBY).")

        
    @ws.setter
    def ws(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("The 'ws' attribute must be of type int or float.")
        self._ws = value   
        
    @type.setter
    def type(self):
        raise ValueError("The 'type' attribute is read-only and cannot be modified.")    

class Parameter_Filiter_High(ParameterFilter):
    """
    表示高通滤波器的参数配置。

    Attributes:
        _type (int): 滤波器类型，固定为 Define_Filiter.High。
        fliter_style (int): 滤波器样式，默认为 -1。
        ws (float): 滤波器的截止频率，必须为 int 或 float 类型。

    Methods:
        type: 获取滤波器类型（只读）。
        ws: 获取或设置滤波器的截止频率。
            - 设置时，必须为 int 或 float 类型，否则抛出 TypeError。
    """
    def __init__():
        self._fliter_style:str=""
        self._ws=0
        self._n=4

    def check_n(self):
        if self._fliter_style=="":
            raise ValueError("The 'fliter_style' attribute must be set before checking 'n'.")

        if self._fliter_style==Define_Filiter.CHEBY:
            if self._n>Define_Filiter._MAX_CHEBY_N:
                raise ValueError("The 'n' attribute must be less than or equal to Define_Filiter._MAX_CHEBY_N.")
    
    @property
    def n(self):
        return self._n
    @property
    def type(self):
        return self._type

    @property
    def ws(self):
        return self._ws
    
    @property
    def fliter_style(self):
        return self._fliter_style
    
    @n.setter
    def n(self, value):
        if not isinstance(value, int):
            raise TypeError("The 'n' attribute must be of type int.")
        self.check_n()
        self._n = value
        
    @fliter_style.setter
    def fliter_style(self, value):
        if value not in Define_Filiter.FILITER_STYLE:
            raise ValueError("The 'fliter_style' attribute must be one of the defined filter styles.Given by (BUTTER,CHEBY)")
    @ws.setter
    def ws(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("The 'ws' attribute must be of type int or float.")
        self._ws = value   
        
    @type.setter
    def type(self):
        raise ValueError("The 'type' attribute is read-only and cannot be modified.")


class Parameter_Filiter_Band(ParameterFilter):
    """
    表示带通滤波器的参数配置。
    
    Attributes:
        _type (int): 滤波器类型，固定为 Define_Filiter.BAND。
        fliter_style (int): 滤波器样式,必须为Define_Filiter.FILITER_STYLE中的一个。
        ws_wp (tuple): 滤波器的截止频率范围，必须为包含两个 int 或 float 的元组。
        n(int): 滤波器的阶数，默认为 4。
    
    Methods:
        type: 获取滤波器类型（只读）。
        ws_wp: 获取或设置滤波器的截止频率范围。
            - 设置时，必须为包含两个 int 或 float 的元组，否则抛出 TypeError。
    """
    def __init__(self):
        self. _type = Define_Filiter._BAND
        self._fliter_style:str=""
        self._ws_wp=()
        self._n=4


    def check_n(self):
        if self._fliter_style=="":
            raise ValueError("The 'fliter_style' attribute must be set before checking 'n'.")
        
        if self._fliter_style==Define_Filiter.CHEBY:
            if self._n>Define_Filiter._MAX_CHEBY_N:
                raise ValueError("The 'n' attribute must be less than or equal to Define_Filiter._MAX_CHEBY_N.")
            
    @property
    def n(self):
        return self._n
    
    @property
    def type(self):
        return self._type

    @property
    def ws_wp(self):
        return self._ws_wp

    @property
    def fliter_style(self):
        return self._fliter_style
    
    @n.setter
    def n(self, value):
        if not isinstance(value, int):
            raise TypeError("The 'n' attribute must be of type int.")
        self.check_n()
        self._n = value
    
    @fliter_style.setter
    def fliter_style(self, value):
        if value not in Define_Filiter.FILITER_STYLE:
            raise ValueError("The 'fliter_style' attribute must be one of the defined filter styles.given by (BUTTER,CHEBY).")
        self._fliter_style = value

    @ws_wp.setter
    def ws_wp(self, value):
        if isinstance(value, tuple):
            if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
                self._ws_wp = value
                return
    
        else:
            raise TypeError("The 'ws_wp' attribute must be a tuple of two int or float values.")
    
    @type.setter
    def type(self):
        raise ValueError("The 'type' attribute is read-only and cannot be modified.")


class Parameter_Filiter_Notch(ParameterFilter):
    """
    表示陷波滤波器的参数配置。
    Attributes:
        _type (ReadOnly->int): 滤波器类型，固定为 Define_Filiter.NOTCH。
        notch_wp (float): 滤波器的截止频率，必须为 int 或 float 类型。
        filter_style (ReadOnly->int): 滤波器样式,固定为 Define_Filiter._NOTCH,此参数是为了其他一般滤波器对齐参数(不影响功能)。
        
    Methods:
        type: 获取滤波器类型（只读）。
        notch_wp: 获取或设置滤波器的截止频率。
        filter_style: 获取滤波器样式（只读）。
            - 设置时，必须为 int 或 float 类型，否则抛出 TypeError。
    """
    _type = Define_Filiter._NOTCH
    _filter_style=Define_Filiter._NOTCH
    _notch_wp=50

    @property
    def type(self):
        return self._type
    @type.setter
    def type(self):
        raise ValueError("The 'type' attribute is read-only and cannot be modified.")
    
    @property
    def notch_wp(self):
        return self._notch_wp

    @notch_wp.setter
    def notch_wp(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("The 'notch_wp' attribute must be of type int or float.")
        self._notch_wp = value

    @property
    def filter_style(self):
        return self._filter_style
    @filter_style.setter
    def filter_style(self):
        raise ValueError("The 'filter_style' attribute is read-only and cannot be modified.")
    
class Test:
    ...