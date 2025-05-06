
from eeg_method.dataset import DataSet
from eeg_method.define import *
from eeg_method.method.trca import TRCA_Method

dataset=DataSet(file_address="G:\Course\MIND_LAB\SSVEP\lesson\dataset1")
dataset.add_dataset_basicinfo(fs=250,info_tuple=(0,1,2,3))

otherinfo=Parameter_DataOtherInfo()
otherinfo.startcut=125
otherinfo.elected_channels=(48, 54, 55, 56, 57, 58, 61, 62, 63)

fliter_band=Parameter_Filiter_Band()
fliter_band.ws_wp=(6,70)
fliter_band.fliter_style=Define_Filiter.BUTTER
fliter_band.n=4

dataset.add_dataset_otherinfo(otherinfo)
dataset.load_data(file_name="S3")
dataset.process_filter(fliter_band)

trca_method=TRCA_Method(dataset=dataset,window_length=0.8)
acc,itr=trca_method.fit()
print(acc,itr)