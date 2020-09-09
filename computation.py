import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


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
    def __init__(self, width=1000, height=1000, space=30, xballs=50, yballs=50, speedrange=2,size=5,frameskip=1,epsilon=0.00001,blocksize=512):
        self.width=np.float32(width)
        self.height=np.float32(height)
        self.xballs=xballs
        self.yballs=yballs
        self.N=np.int32(xballs*yballs) #CUDA takes in 32 bit ints
        self.speedrange=speedrange
        self.size=np.float32(size)
        self.space=space
        self.frameskip=frameskip
        self.epsilon=np.float32(epsilon)
        self.blocksize=blocksize
        iterations = int(self.N*(self.N-1)/2)
        self.gridsize = int(np.ceil(iterations/self.blocksize));
        print(f"There are {self.N} balls --> {iterations} iterations... Meaning {self.gridsize} blocks of size {self.blocksize}")
        self.v = np.zeros((2,self.N), dtype=np.float32)
        self.pos = np.zeros((2,self.N), dtype=np.float32)
        print(f"coords have shape {np.shape(self.pos)}")
        self.module=getmodule("boltzmann.cu")
        self.cstep = self.module.get_function("step_kernel")

        for i in range(0, yballs):
            for j in range(0, xballs):
                initx = (((j + 1) * self.space) - 1)-width
                inity = ((-1 * (i + 1) * self.space) + 1)+height
                self.pos[:, xballs * i + j] = [initx, inity];
                self.v[:,xballs*i+j] = [np.random.uniform(-speedrange,speedrange),np.random.uniform(-speedrange,speedrange)];
                print(f"VEL: {self.v[:,xballs*i+j]}")


#double *posx, double *posy, double *vx, double *vy, int N, double size, double epsilon, double width, double height
    def cudastep(self):
        self.cstep(drv.InOut(self.pos[0]), drv.InOut(self.pos[1]), drv.InOut(self.v[0]), drv.InOut(self.v[1])
                            ,self.N,self.size,self.epsilon,self.width,self.height, block=(self.blocksize,1,1), grid=(self.gridsize,1))
