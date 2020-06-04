#!/usr/bin/env python
#{{{ IMPORTS

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import math
from PIL import Image

#}}}
#{{{WINDOW INIT

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.TRUE);
#glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
altura = 768
largura = 768
window = glfw.create_window(largura, altura, "T2", None, None)
glfw.make_context_current(window)
glfw.set_window_pos(window, int(1366/4), 0)

#}}}
#{{{GLSL

vertex_code = """
        attribute vec3 position;
        attribute vec2 texture_coord;
        varying vec2 out_texture;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main(){
            gl_Position = projection * view * model * vec4(position,1.0);
            out_texture = vec2(texture_coord);
        }
        """

fragment_code = """
        uniform vec4 color;
        varying vec2 out_texture;
        uniform sampler2D samplerTexture;

        void main(){
            vec4 texture = texture2D(samplerTexture, out_texture);
            gl_FragColor = texture;
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
    raise RuntimeError("Erro de compilacao do Vertex Shader")

glCompileShader(fragment)
if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(fragment).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Fragment Shader")

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

#}}}
#{{{ AUX

# Retirado do exemplo de aula
def load_model_from_file(filename):
    objects = {}
    vertices = []
    texture_coords = []
    faces = []

    material = None

    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue

        if values[0] == 'v':
            vertices.append(values[1:4])
        elif values[0] == 'vt':
            texture_coords.append(values[1:3])
        elif values[0] in ('usemtl', 'usemat'):
            material = values[1]
        elif values[0] == 'f':
            face = []
            face_texture = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0]))
                if len(w) >= 2 and len(w[1]) > 0:
                    face_texture.append(int(w[1]))
                else:
                    face_texture.append(0)

            faces.append((face, face_texture, material))

    model = {}
    model['vertices'] = vertices
    model['texture'] = texture_coords
    model['faces'] = faces

    return model

def load_texture_from_file(texture_id, img_textura):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    img = Image.open(img_textura)
    img_width, img_height = img.size
    image_data = img.convert("RGBA").tobytes("raw", "RGBA", 0, -1)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_width, img_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)

#}}}
#{{{ RANDOM GLOBAL VARIABLES

texture_count = 0
modelos = {}

#}}}
#{{{ LOAD MODEL AND TEXTURES

# Enable textures, allocate memory
glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
glEnable(GL_BLEND)
glDisable(GL_CULL_FACE)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glEnable(GL_LINE_SMOOTH)
glEnable(GL_TEXTURE_2D)
qtd_texturas = 10
textures = glGenTextures(qtd_texturas)

vertices_list = []
textures_coord_list = []

# Carrega o terreno
modelo = load_model_from_file('models/terrain/terreno.obj')
modelos['terreno'] = {}
modelos['terreno']['n_texturas'] = 1
modelos['terreno']['start'] = len(vertices_list)
print('Processando modelo terreno.obj')
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
modelos['terreno']['size'] = len(vertices_list) - modelos['terreno']['start']
modelos['terreno']['texture_id'] = texture_count
load_texture_from_file(modelos['terreno']['texture_id'], 'models/terrain/chess.jpg')
texture_count += 1

# Carrega a casa
modelo = load_model_from_file('models/house/medieval house.obj')
modelos['casa'] = {}
modelos['casa']['n_texturas'] = 1
modelos['casa']['start'] = len(vertices_list)
print('Processando modelo casa.obj')
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
modelos['casa']['size'] = len(vertices_list) - modelos['casa']['start']
modelos['casa']['texture_id'] = texture_count
load_texture_from_file(modelos['casa']['texture_id'], 'models/house/house2.png')
texture_count += 1

print(modelos)

# Request a buffer slot from GPU
buffer = glGenBuffers(2)

vertices = np.zeros(len(vertices_list), [("position", np.float32, 3)])
vertices['position'] = vertices_list

# Upload data
glBindBuffer(GL_ARRAY_BUFFER, buffer[0])
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
stride = vertices.strides[0]
offset = ctypes.c_void_p(0)
loc_vertices = glGetAttribLocation(program, "position")
glEnableVertexAttribArray(loc_vertices)
glVertexAttribPointer(loc_vertices, 3, GL_FLOAT, False, stride, offset)

textures = np.zeros(len(textures_coord_list), [("position", np.float32, 2)]) # duas coordenadas
textures['position'] = textures_coord_list

# Upload data
glBindBuffer(GL_ARRAY_BUFFER, buffer[1])
glBufferData(GL_ARRAY_BUFFER, textures.nbytes, textures, GL_STATIC_DRAW)
stride = textures.strides[0]
offset = ctypes.c_void_p(0)
loc_texture_coord = glGetAttribLocation(program, "texture_coord")
glEnableVertexAttribArray(loc_texture_coord)
glVertexAttribPointer(loc_texture_coord, 2, GL_FLOAT, False, stride, offset)

#}}}
#{{{ INPUT EVENTS

cameraPos   = glm.vec3(0.0,  0.0,  1.0);
cameraFront = glm.vec3(0.0,  0.0, -1.0);
cameraUp    = glm.vec3(0.0,  1.0,  0.0);

wireframe = False

def key_event(window,key,scancode,action,mods):
    global cameraPos, cameraFront, cameraUp, wireframe

    # quit simulation
    if (key == glfw.KEY_Q or key == glfw.KEY_ESCAPE) and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    cameraSpeed = 0.3
    if key == glfw.KEY_W and (action == glfw.PRESS or
                              action == glfw.REPEAT):
        cameraPos += cameraSpeed * cameraFront

    if key == glfw.KEY_S and (action == glfw.PRESS or
                              action == glfw.REPEAT):
        cameraPos -= cameraSpeed * cameraFront

    if key == glfw.KEY_A and (action == glfw.PRESS or
                              action == glfw.REPEAT):
        cameraPos -= glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed

    if key == glfw.KEY_D and (action == glfw.PRESS or
                              action == glfw.REPEAT):
        cameraPos += glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed

    if key == glfw.KEY_P and action == glfw.PRESS:
        wireframe = not wireframe

firstMouse = True
yaw = -90.0
pitch = 0.0
lastX =  largura/2
lastY =  altura/2

def mouse_event(window, xpos, ypos):
    global firstMouse, cameraFront, yaw, pitch, lastX, lastY
    if firstMouse:
        lastX = xpos
        lastY = ypos
        firstMouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos

    sensitivity = 0.15
    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset;
    pitch += yoffset;

    if pitch >= 90.0: pitch = 90.0
    if pitch <= -90.0: pitch = -90.0

    front = glm.vec3()
    front.x = math.cos(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    front.y = math.sin(glm.radians(pitch))
    front.z = math.sin(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    cameraFront = glm.normalize(front)

glfw.set_key_callback(window,key_event)
glfw.set_cursor_pos_callback(window, mouse_event)
# Disable the cursor, making it always centered
glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

#}}}
#{{{ DRAW FUNCTIONS

def desenha_terreno():
    angle = 0.0;
    r_x = 0.0; r_y = 0.0; r_z = 1.0;
    t_x = 0.0; t_y = -1.01; t_z = 0.0;
    s_x = 50.0; s_y = 50.0; s_z = 50.0;
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['terreno']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['terreno']['start'], modelos['terreno']['size'])

def desenha_casa():
    angle = 0.0;
    r_x = 0.0; r_y = 0.0; r_z = 1.0;
    t_x = 0.0; t_y = -1.01; t_z = -10.0;
    s_x = 0.5; s_y = 0.5; s_z = 0.5;
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['casa']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['casa']['start'], modelos['casa']['size'])

#}}}
#{{{ MODEL VIEW PROJECTION

def model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z):
    angle = math.radians(angle)
    matrix_transform = glm.mat4(1.0)
    matrix_transform = glm.rotate(matrix_transform, angle, glm.vec3(r_x, r_y, r_z))
    matrix_transform = glm.translate(matrix_transform, glm.vec3(t_x, t_y, t_z))
    matrix_transform = glm.scale(matrix_transform, glm.vec3(s_x, s_y, s_z))
    matrix_transform = np.array(matrix_transform).T
    return matrix_transform

def view():
    global cameraPos, cameraFront, cameraUp
    mat_view = glm.lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    mat_view = np.array(mat_view)
    return mat_view

def projection():
    global altura, largura
    mat_projection = glm.perspective(glm.radians(90.0), largura/altura, 0.1, 1000.0)
    mat_projection = np.array(mat_projection)
    return mat_projection
#}}}
#{{{ LOOP

glfw.show_window(window)
glfw.set_cursor_pos(window, largura/2, altura/2)

glEnable(GL_DEPTH_TEST)

while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(0.68, 0.85, 0.9, 1.0)

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if wireframe else GL_FILL)

    desenha_terreno()
    desenha_casa()

    mat_view = view()
    loc_view = glGetUniformLocation(program, "view")
    glUniformMatrix4fv(loc_view, 1, GL_FALSE, mat_view)

    mat_projection = projection()
    loc_projection = glGetUniformLocation(program, "projection")
    glUniformMatrix4fv(loc_projection, 1, GL_FALSE, mat_projection)

    glfw.swap_buffers(window)

glfw.terminate()
#}}}
