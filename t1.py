#   To test it:
#   python -m venv project_dir
#   pip install glfw numpy pyopengl
#   python t1.py
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import math

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
    #vertices['position'][i] = [ r_spring*math.sin(((len(vertices)-i)/len(vertices))*2*math.pi)*math.sin(((len(vertices)-i)/len(vertices))*20*math.pi), y_attenuation_value*(i-(len(vertices)/2)) ]

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

# rotation
t_x = 0.0
t_y = 0.0
angle = 0.0


def key_event(window,key,scancode,action,mods):
    global x_inc, y_inc, r_inc
    
    if key == 263: x_inc -= 0.0001
    if key == 262: x_inc += 0.0001
    
    if key == 265: y_inc += 0.0001
    if key == 264: y_inc -= 0.0001
    
    if key == 65: r_inc += 0.1
    if key == 83: r_inc -= 0.1
    
    if key == 81: glfw.set_window_should_close(window, True)
    #print(key)
    
glfw.set_key_callback(window,key_event)

glfw.show_window(window)


# Loop 


def matrix_mult(a,b):
    m_a = a.reshape(4,4)
    m_b = b.reshape(4,4)
    m_c = np.dot(m_a,m_b)
    c = m_c.reshape(1,16)
    return c


while not glfw.window_should_close(window):

    t_x += x_inc
    t_y += y_inc
    angle += r_inc
    
    c = math.cos( math.radians(angle) )
    s = math.sin( math.radians(angle) )
    
    glfw.poll_events() 
    
    
    glClear(GL_COLOR_BUFFER_BIT) 
    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    # Applying transformations
    
    mat_rotation = np.array([  c  , -s , 0.0, 0.0, 
    s  , c  , 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 
    0.0, 0.0, 0.0, 1.0], np.float32)
    
    mat_translation = np.array([  1.0, 0.0, 0.0, t_x, 
    0.0, 1.0, 0.0, t_y, 
    0.0, 0.0, 1.0, 0.0, 
    0.0, 0.0, 0.0, 1.0], np.float32)
    
    mat_transform = matrix_mult(mat_translation,mat_rotation)    
    
    loc = glGetUniformLocation(program, "mat")
    glUniformMatrix4fv(loc, 1, GL_TRUE, mat_transform)
    
    #Draw 
    
    glDrawArrays(GL_LINE_STRIP, 0, len(vertices))
    
    glfw.swap_buffers(window)
    
glfw.terminate()
