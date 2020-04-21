#   To test it:
#   python -m venv project_dir
#   pip install glfw numpy pyopengl
#   python t1.py

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import math
import random

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE);
window = glfw.create_window(800, 800, "T1", None, None)
glfw.make_context_current(window)

# GLSL (OpenGL Shading Language)

# Vertex Shader
vertex_code = """
    attribute vec2 position;
    uniform mat4 mat;
    void main(){
    gl_Position = mat * vec4(position,0.0,1.0);
    }
    """

# Fragment Shader
fragment_code = """
    void main(){
    gl_FragColor = vec4(1.0,0.0,0.0,1.0);
    }
    """

# Request a program and shader slots from GPU
program  = glCreateProgram()
vertex   = glCreateShader(GL_VERTEX_SHADER)
fragment = glCreateShader(GL_FRAGMENT_SHADER)

# Set shaders source
glShaderSource(vertex, vertex_code)
glShaderSource(fragment, fragment_code)

# Compile shaders
glCompileShader(vertex)
if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(vertex).decode()
    print(error)
    raise RuntimeError("Error during Vertex Shader compilation.")

glCompileShader(fragment)
if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(fragment).decode()
    print(error)
    raise RuntimeError("Error during Fragment Shader compilation.")

# Attach shader objects to the program
glAttachShader(program, vertex)
glAttachShader(program, fragment)

# Build program
glLinkProgram(program)
if not glGetProgramiv(program, GL_LINK_STATUS):
    print(glGetProgramInfoLog(program))
    raise RuntimeError('Linking error')

# Make program the default program
glUseProgram(program)

vertices = np.zeros(400, [("position", np.float32, 2)])
n_revolutions = 5
r_spring = 0.05
y_attenuation_value = 0.001
for i in range(len(vertices)):
    vertices['position'][i] = [ r_spring*math.cos((i/(len(vertices)/n_revolutions))*2*math.pi), y_attenuation_value*(i-(len(vertices)/2)) ]
    # versao macacos do artico abaixo
    # vertices['position'][i] = [ r_spring*math.sin(((len(vertices)-i)/len(vertices))*2*math.pi)*math.sin(((len(vertices)-i)/len(vertices))*20*math.pi), y_attenuation_value*(i-(len(vertices)/2)) ]

# Request a buffer slot from GPU
buffer = glGenBuffers(1)

# Make this buffer the default one
glBindBuffer(GL_ARRAY_BUFFER, buffer)

# Upload data
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
glBindBuffer(GL_ARRAY_BUFFER, buffer)

# Bind the position attribute
stride = vertices.strides[0]
offset = ctypes.c_void_p(0)

loc = glGetAttribLocation(program, "position")
glEnableVertexAttribArray(loc)

glVertexAttribPointer(loc, 2, GL_FLOAT, False, stride, offset)

# translation
x_inc = 0.0
y_inc = 0.0
r_inc = 0.0

# delta position
t_x = 0.0
t_y = 0.0
a_x = a_y = 0.0 #initial acceleration
v_x = v_y = 0.0 #initial velocity

# rotation
angle = 0.0

# compression
com = 0.0
is_pressed = False

def key_event(window, key, scancode, action, mods):
    global com, a_y, v_y, v_x, is_pressed
    if key == glfw.KEY_DOWN:
        is_pressed = True
        if com < 3:
            com += 0.01

        if action == glfw.RELEASE:
            v_y = 1.1 * com # Make the "jump" proportional to the compression
            v_x = 1.1 * com
            if (math.floor(global_time*100000)%2) == 1: # random direction to jump
                v_x *= -1
            a_y = -1
            is_pressed = False

    # quit simulation
    if key == glfw.KEY_Q:
        glfw.set_window_should_close(window, True)

glfw.set_key_callback(window, key_event)

glfw.show_window(window)

def matrix_mult(a,b):
    m_a = a.reshape(4,4)
    m_b = b.reshape(4,4)
    m_c = np.dot(m_a,m_b)
    c = m_c.reshape(1,16)
    return c

# Loop
##  setting semi-fixed time step variables
global_time = 0.0
dt = 1.0/60 # defining minimal refresh rate

cur_time = glfw.get_time()

while not glfw.window_should_close(window):
    time = glfw.get_time()
    frame_time = time - cur_time
    cur_time = time

    # event input handling
    glfw.poll_events()

    while frame_time > 0.0:
        delta = min(frame_time,dt)

        frame_time -= delta
        global_time += delta

        #t_x += x_inc
        #t_y += y_inc
        #angle += r_inc

        # getting translation by semi-implicit euler method
        v_x += a_x*delta # velocity changes due to acceleration values
        v_y += a_y*delta

        t_x += v_x*delta # t_x,y represents position
        t_y += v_y*delta

        if t_y < 0.0:
            a_y = 0.0
            v_y = 0.0
            v_x = 0.0

        c = math.cos( math.radians(angle) )
        s = math.sin( math.radians(angle) )

        # Applying transformations
        mat_rotation = np.array(
        [ c,  -s, 0.0, 0.0,
          s,   c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0], np.float32)

        mat_translation = np.array(
        [1.0, 0.0, 0.0, t_x,
         0.0, 1.0, 0.0, t_y,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0], np.float32)

        _mat_transform = matrix_mult(mat_translation, mat_rotation)

        mat_compress = np.array(
        [1.0, 0.0, 0.0, 0.0,
         0.0, 1.0/(com+1.0), 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0], np.float32)

        mat_transform = matrix_mult(mat_compress, _mat_transform)

        loc = glGetUniformLocation(program, "mat")
        glUniformMatrix4fv(loc, 1, GL_TRUE, mat_transform)

        # cleaning color buffers
        glClear(GL_COLOR_BUFFER_BIT)
        glClearColor(1.0, 1.0, 1.0, 1.0)

        #Draw
        glDrawArrays(GL_LINE_STRIP, 0, len(vertices))
        glfw.swap_interval(1)
        glfw.swap_buffers(window)
        print("t=", global_time)
        print("v=", v_y)
        print("a=", a_y)
        print(f"com = {com}")

        # De-compress the spring
        if not is_pressed and com > 0.0:
            com -= 0.1

glfw.terminate()
