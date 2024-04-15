import os
import sys
from predict_miou import func
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
canvas="black"
model="fcn" 
grid_size=7


data_root=""

output_dir=""

assert model in ["fcn", "resnet18", "resnet34"]
if os.path.exists(output_dir) == False:
    print("dir is not exist")
    sys.exit()


mode=7
func(canvas=canvas,grid_size=grid_size,translate=1,mode=mode,data_root=data_root,output_dir=output_dir,model_=model)



mode=8
func(canvas=canvas,grid_size=grid_size,translate=1,mode=mode,data_root=data_root,output_dir=output_dir,model_=model)

for translate in [1,3,5,7,9]:
    mode = 2
    func(canvas=canvas,grid_size=grid_size,translate=translate,mode=mode,data_root=data_root,output_dir=output_dir,model_=model)

mode=1
func(canvas=canvas,grid_size=grid_size,translate=1,mode=mode,data_root=data_root,output_dir=output_dir,model_=model)


