import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from constants import *

def mag(x):
    return np.sqrt(sum(i**2 for i in x))
def magsq(x):
    return (sum(i**2 for i in x))

def getmodule(path):
    try:
        f = open(path, "r")
    except:
        print(f"could not open file at {path}")
        return None
    source = f.read();
    f.close()
    return SourceModule(source)


class Computation:
    def __init__(self):
        self.v = np.zeros((2,BALLCOUNT), dtype=np.float32)
        self.pos = np.zeros((2,BALLCOUNT), dtype=np.float32)
        self.module = getmodule("boltzmann.cu")
        self.cstep = self.module.get_function("step")
        self.ccheck = self.module.get_function("collisioncheck")
        for i in range(0, YBALLS):
            for j in range(0, XBALLS):
                initx = (((j + 1) * SPACE) - 1)-WIDTH
                inity = ((-1 * (i + 1) * SPACE) + 1)+HEIGHT
                self.pos[:, XBALLS * i + j] = [initx, inity];
                self.v[:,XBALLS*i+j] = [np.random.uniform(-INITSPEED,INITSPEED),np.random.uniform(-INITSPEED,INITSPEED)];

    def cudastep(self):
        #Python's integers aren't 32 bit but CUDA wants that
        N = np.int32(BALLCOUNT)

        grid_size = int((N - 1 + BLOCKSIZE) / BLOCKSIZE)
        self.cstep(drv.InOut(self.pos[0]), drv.InOut(self.pos[1]), drv.InOut(self.v[0]), drv.InOut(self.v[1]), N,
                   np.float32(SIZE), np.float32(EPSILON), np.float32(WIDTH), np.float32(HEIGHT), block=(BLOCKSIZE, 1, 1), grid=(grid_size,1));

        grid_size = int((((N - 1) * (N / 2)) - 1 + BLOCKSIZE) / BLOCKSIZE);
        self.ccheck(drv.InOut(self.pos[0]), drv.InOut(self.pos[1]), drv.InOut(self.v[0]), drv.InOut(self.v[1]), N,
                    np.float32(SIZE), np.float32(WIDTH), np.float32(HEIGHT), block=(BLOCKSIZE, 1, 1), grid=(grid_size, 1));