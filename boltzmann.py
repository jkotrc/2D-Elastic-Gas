#MAIN method and graphics
try:
    from OpenGL.GL import *
    from OpenGL import GLU

    import OpenGL.GL.shaders
except:
    print("OpenGL wrapper for python not found")
import glfw
import numpy as np
from computation import Computation
from constants import *

class Graphics:
    def __init__(self,width,height):
        if not glfw.init():
            print("GLFW Failed to initialize!")
        self.window = glfw.create_window(width, height, "Boltzmann", None, None);
        glfw.make_context_current(self.window)
        self.windowsizechanged=False
        glfw.set_window_size_callback(self.window, self.resizewindow)
        self.program = self.loadShaders("vertex.glsl", "fragment.glsl")
        glUseProgram(self.program)
        glUniform1i(glGetUniformLocation(self.program, "WIDTH"), WIDTH)
        glUniform1i(glGetUniformLocation(self.program, "HEIGHT"), HEIGHT)

        self.width=width
        self.height=height
        self.comp = Computation()
        self.points = np.array(self.comp.pos.reshape(-1,order='F'), dtype=np.float32)
        self.graphicsinit()


    def resizewindow(self,w,h,a):
        self.windowsizechanged=True

    def graphicsinit(self):
        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, self.points.itemsize * self.points.size, self.points, GL_STATIC_DRAW)
        position = glGetAttribLocation(self.program, "position")
        glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(position)
        glClearColor(0.3, 0.3, 0.3, 1.0)
        glEnable(GL_POINT_SMOOTH)
        glPointSize(SIZE/2)




    def render(self):
        for i in range (0, FRAMESKIP):
            self.comp.cudastep();

        self.points = self.comp.pos.reshape(-1,order='F')
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)
        glBufferData(GL_ARRAY_BUFFER, self.points.itemsize * self.points.size, self.points, GL_STATIC_DRAW)
        glDrawArrays(GL_POINTS, 0, int(self.points.size / 2))
        glfw.swap_buffers(self.window)


    def mainloop(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            if self.windowsizechanged == True:
                w,h = glfw.get_framebuffer_size(self.window);
                WIDTH = w
                HEIGHT = h
                glUseProgram(self.program)
                glUniform1i(glGetUniformLocation(self.program, "WIDTH"), WIDTH)
                glUniform1i(glGetUniformLocation(self.program, "HEIGHT"), HEIGHT)
                self.windowsizechanged=False
            self.render()

        glfw.terminate()

    def loadShaders(self, vertpath, fragpath):
        vertexshader=glCreateShader(GL_VERTEX_SHADER)
        fragmentshader=glCreateShader(GL_FRAGMENT_SHADER)
        fragfile = open(fragpath, "r")
        vertfile = open(vertpath, "r")
        fragsource = fragfile.read()
        fragfile.close()
        vertsource = vertfile.read()
        vertfile.close()
        shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertsource, GL_VERTEX_SHADER),
                                                  OpenGL.GL.shaders.compileShader(fragsource, GL_FRAGMENT_SHADER))
        return shader


if __name__ == "__main__":
    #TODO: separate the computation class from the graphics class so it can be e.g. simulated off screen or saved to a file, etc.
    g=Graphics(WIDTH, HEIGHT)
    g.mainloop();








