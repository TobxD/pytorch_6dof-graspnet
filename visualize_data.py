import time
from options.train_options import TrainOptions
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
    cnt = 1
    for i, data in enumerate(dataset):
        #pprint({
        #    k: data[k].shape for k in data
        #})
        if np.random.rand() > 0.5:
            ind = np.argmin(data["quality"])
        else:
            ind = 0
        pc = data["pc"][ind]
        print("quality:", data["quality"][ind])
        mlab.figure(bgcolor=(1, 1, 1))
        grasp = data["grasp_rt"][ind]
        if grasp.shape[-1] == 16:
            grasp = grasp.reshape(4,4)
            draw_scene(pc, grasps=[grasp])
        else:
            draw_scene(pc)
            red = np.array([255, 0, 0])
            green = np.array([0, 255, 0])
            qual = (data["quality"][ind]/2 + 0.5)
            colors = qual * green + (1-qual) * red
            draw_scene(grasp, pc_color=colors)
        print('close the window to continue to next object . . .')
        mlab.show()
        cnt -= 1
        if cnt == 0:
            break

if __name__ == '__main__':
    main()
