import data
import argparse
from model import EDSR
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="./data/General-100")
parser.add_argument("--imgsize",default=64,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=16,type=int)
parser.add_argument("--savedir",default='saved_models')
parser.add_argument("--iterations",default=500,type=int)
args = parser.parse_args()
img_path = ""
target_path = ""
train_csv_path = ""
test_csv_path = ""
#data.load_dataset(r'D:\python\EDSR-Tensorflow-master\dataset\med_dataset',args.imgsize)
if args.imgsize % args.scale != 0:
    print(f"Image size {args.imgsize} is not evenly divisible by scale {args.scale}")
    exit()
#down_size = args.imgsize//args.scale
network = EDSR(args.imgsize,args.layers,args.featuresize,args.scale)
network.set_data_fn(data.train_generator,
                    (args.batchsize, img_path, target_path, train_csv_path, args.imgsize, args.scale),
                    (args.batchsize, img_path, target_path, test_csv_path, args.imgsize, args.scale))
network.train(args.iterations,args.savedir)
