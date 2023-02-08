import time
from options.train_options import TrainOptions
if True:
    from acronym_data import DataLoader
else:
    from data import DataLoader
from models import create_model
from utils.writer import Writer
from test import run_test
import threading
from pprint import pprint
import mayavi.mlab as mlab
from utils.visualization_utils import *
import mayavi.mlab as mlab

def main():
    opt = TrainOptions().parse()
    if opt == None:
        return

    dataset = DataLoader(opt)
    dataset_size = len(dataset) * opt.num_grasps_per_object
    cnt = 5
    for i, data in enumerate(dataset):
        pprint({
            k: data[k].shape for k in data
        })
        pc = data["pc"][0]
        grasp = data["grasp_rt"][0].reshape(4,4)
        print(grasp.shape)
        mlab.figure(bgcolor=(1, 1, 1))
        draw_scene(pc, grasps=[grasp])
        print('close the window to continue to next object . . .')
        mlab.show()
        cnt -= 1
        if cnt == 0:
            break

if __name__ == '__main__':
    main()
