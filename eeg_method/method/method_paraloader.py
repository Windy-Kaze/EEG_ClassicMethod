


class Flag_counter:
    __count=-1
    @classmethod
    def count(cls):
        cls.__count+=1
        return cls.__count
    
class CCA_parameter_loader:
    """
    CCA方法的参数加载器类。
    """
    def __init__(self, frq_tuple: tuple, num_harmony=4, window_length=0.8) -> None:
        self.frq_tuple = frq_tuple
        self.num_harmony = num_harmony
        self.window_length = window_length
        self.__parameter_checker()  
 
    def __parameter_checker(self) -> None:
        """
        检查参数的有效性。
        """
        if not isinstance(self.frq_tuple, tuple):
            raise TypeError("frq_tuple must be tuple")
        
        if not all(item>0 for item in self.frq_tuple):
            raise ValueError("every elements which infrq_tuple must be >0") 
        if self.window_length<=0 or self.window_length>1:
            raise ValueError("window_length must be in (0,1]")
        if not isinstance(self.num_harmony, int):
            raise TypeError("num_harmony must be int")
    

        
        
        
        
    