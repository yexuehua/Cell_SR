import pandas as pd
import os
import numpy as np

top_path = r"C:\Users\212774000\Documents\python\gitclone\SR\Cell_SR\data"
lr_names = os.listdir(os.path.join(top_path,"overlay256"))
lr_names.sort(key=lambda x:int(x[:-12]))
lr_lists = [os.path.join("overlay256",i) for i in lr_names]
hr_names = os.listdir(os.path.join(top_path,"overlay512"))
hr_names.sort(key=lambda x:int(x[:-12]))
hr_lists = [os.path.join("overlay512",i) for i in hr_names]
df = pd.DataFrame({"overlay256":lr_lists,
                   "overlay512":hr_lists})
df.to_csv("data.csv")
ndf = pd.read_csv("data.csv")
print(list(ndf["overlay256"]))