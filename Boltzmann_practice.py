import numpy as np
import cupy as cp
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import time





#max block dimensions: (1024,1024,64)
#max grid dimensions (2147483647,65535,65535)
#1024 threads per block
#2048 threads per multiprocessor
#6 multiprocessors
def printstats():
    (free, total) = drv.mem_get_info()
    print("Global memory occupancy:%f%% free" % (free * 100 / total))

    for devicenum in range(drv.Device.count()):
        device = drv.Device(devicenum)
        attrs = device.get_attributes()

        # Beyond this point is just pretty printing
        print("\n===Attributes for device %d" % devicenum)
        for (key, value) in attrs.items():
            print("%s:%s" % (str(key), str(value)))



def getmodule(path):
    f = open(path, "r")
    source = f.read();
    f.close()
    return SourceModule(source)


if __name__ == "__main__":
    print("trying to do a thing")
    module = getmodule("experimentation.cu")
    #test = module.get_function("printIndex");
    test = module.get_function("add1");
    N=np.int32(6000)
    dummy = np.zeros((N), dtype=np.int32)
    test(drv.InOut(dummy),N, block=(512,1,1));

    print(f"{0} should correspond to {np.array(np.where(dummy == 0)).size}");



#add = mod.get_function("add");
#add(drv.Out(result), drv.In(a), drv.In(b), block=(1000,1,1), grid=(1,1))
#kernel <<<numBlocks, threadsPerBlock>>>
#this is where cl.exe is
#C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.25.28610\bin\Hostx64\x64
#time.time() % 60 = seconds








