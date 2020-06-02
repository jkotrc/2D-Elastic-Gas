#Some constants for the simulation
HEIGHT=1000
WIDTH=1000

#make an XBALLS by YBALLS grid of balls, SPACE apart
SPACE=30
XBALLS=60
YBALLS=60

#random speeds with magnitude between 0 and 10
INITSPEED=10
SIZE=5
BALLCOUNT=int(XBALLS*YBALLS)

#render every nth frame. If the epsilon value is low, not every frame has to be rendered because the frames look too similar
#this is not (yet) efficient as the state of the gas has to be pointlessly sent back and forth between the GPU without modification
FRAMESKIP=2

EPSILON=0.1

#should be a multiple of the warp size (32 on a 1050Ti) but should be smaller or equal to the number of balls
#512 is often the optimum
BLOCKSIZE=32