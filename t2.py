#!/usr/bin/env python
#{{{ IMPORTS

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import math
from PIL import Image
import simpleaudio as sa


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

deer_angle = 0
cameraSpeed = 5
sensitivity = 0.15

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

# Skybox?
modelo = load_model_from_file('models/skybox/untitled.obj')
modelos['skybox'] = {}
modelos['skybox']['n_texturas'] = 1
modelos['skybox']['start'] = len(vertices_list)
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
modelos['skybox']['size'] = len(vertices_list) - modelos['skybox']['start']
modelos['skybox']['texture_id'] = texture_count
load_texture_from_file(modelos['skybox']['texture_id'], 'models/skybox/blue.png')
texture_count += 1


# Gramado
modelo = load_model_from_file('models/terrain/terreno.obj')
modelos['terrain'] = {}
modelos['terrain']['n_texturas'] = 1
modelos['terrain']['start'] = len(vertices_list)
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
modelos['terrain']['size'] = len(vertices_list) - modelos['terrain']['start']
modelos['terrain']['texture_id'] = texture_count
load_texture_from_file(modelos['terrain']['texture_id'], 'models/terrain/grass.jpg')
texture_count += 1

# Rua
modelo = load_model_from_file('models/terrain/terreno.obj')
modelos['road'] = {}
modelos['road']['n_texturas'] = 1
modelos['road']['start'] = len(vertices_list)
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
modelos['road']['size'] = len(vertices_list) - modelos['road']['start']
modelos['road']['texture_id'] = texture_count
load_texture_from_file(modelos['road']['texture_id'], 'models/terrain/road2.jpg')
texture_count += 1

# Casa
modelo = load_model_from_file('models/house/untitled.obj')
modelos['house'] = {}
modelos['house']['n_texturas'] = 1
modelos['house']['start'] = len(vertices_list)
print('Processando modelo house.obj')
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
modelos['house']['size'] = len(vertices_list) - modelos['house']['start']
modelos['house']['texture_id'] = texture_count
load_texture_from_file(modelos['house']['texture_id'], 'models/house/Hut_Low_lambert1_AlbedoTransparency.png')
texture_count += 1

# Denis
modelo = load_model_from_file('models/person/denis.obj')
modelos['person'] = {}
modelos['person']['n_texturas'] = 1
modelos['person']['start'] = len(vertices_list)
print('Processando modelo person.obj')
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
modelos['person']['size'] = len(vertices_list) - modelos['person']['start']
modelos['person']['texture_id'] = texture_count
load_texture_from_file(modelos['person']['texture_id'], 'models/person/denis.jpg')
texture_count += 1

# Knuckles
modelo = load_model_from_file('models/uganda/Knuckles.obj')
modelos['uganda_knuckles'] = {}
modelos['uganda_knuckles']['n_texturas'] = 1
modelos['uganda_knuckles']['start'] = len(vertices_list)
print('Processando modelo uganda_knuckles.obj')
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
modelos['uganda_knuckles']['size'] = len(vertices_list) - modelos['uganda_knuckles']['start']
modelos['uganda_knuckles']['texture_id'] = texture_count
load_texture_from_file(modelos['uganda_knuckles']['texture_id'], 'models/uganda/Knuckles_Texture.png')
texture_count += 1

# Statue
modelo = load_model_from_file('models/statue2/untitled.obj')
modelos['statue'] = {}
modelos['statue']['n_texturas'] = 1
modelos['statue']['start'] = len(vertices_list)
print('Processando modelo statue.obj')
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
modelos['statue']['size'] = len(vertices_list) - modelos['statue']['start']
modelos['statue']['texture_id'] = texture_count
load_texture_from_file(modelos['statue']['texture_id'], 'models/statue2/DavidFixedDiff.jpg')
texture_count += 1

# Tree
modelo = load_model_from_file('models/tree/tree_X12_+X1_Rock_Pack.obj')
modelos['tree'] = {}
modelos['tree']['n_texturas'] = 1
modelos['tree']['start'] = len(vertices_list)
print('Processando modelo tree.obj')
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
modelos['tree']['size'] = len(vertices_list) - modelos['tree']['start']
modelos['tree']['texture_id'] = texture_count
load_texture_from_file(modelos['tree']['texture_id'], 'models/tree/_6_tree.png')
texture_count += 1

# Deer
modelo = load_model_from_file('models/deer/untitled.obj')
modelos['deer'] = {}
modelos['deer']['n_texturas'] = 1
modelos['deer']['start'] = len(vertices_list)
print('Processando modelo deer.obj')
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
modelos['deer']['size'] = len(vertices_list) - modelos['deer']['start']
modelos['deer']['texture_id'] = texture_count
load_texture_from_file(modelos['deer']['texture_id'], 'models/deer/Diffuse.jpg')
texture_count += 1

# Bench
modelo = load_model_from_file('models/bench/uploads_files_839016_OutdoorParkBenches(1).obj')
modelos['bench'] = {}
modelos['bench']['n_texturas'] = 2
modelos['bench']['start'] = len(vertices_list)
print('Processando modelo bench.obj')
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
modelos['bench']['size'] = len(vertices_list) - modelos['bench']['start']
modelos['bench']['texture_id'] = texture_count
load_texture_from_file(modelos['bench']['texture_id'], 'models/bench/OutdoorParkBenches_woods_BaseColor.png')
load_texture_from_file(modelos['bench']['texture_id']+1, 'models/bench/OutdoorParkBenches_Steel_BaseColor.png')
texture_count += 2

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

cameraPos   = glm.vec3(100.0,  100.0,  100.0);
cameraFront = glm.vec3(0.0,  0.0, -1.0);
cameraUp    = glm.vec3(0.0,  1.0,  0.0);

def skybox(pos):
    if -1024 < pos[0] < 1024 and 10 < pos[1] < 1024 and -1024 < pos[2] < 1024:
        return True
    return False

wireframe = False

def key_event(window,key,scancode,action,mods):
    global cameraPos, cameraFront, cameraUp
    global wireframe, scale, cameraSpeed, sensitivity

    # quit simulation with <ESC> or Q
    if (key == glfw.KEY_Q or key == glfw.KEY_ESCAPE) and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
    if key == glfw.KEY_W and (action == glfw.PRESS or action == glfw.REPEAT):
        if skybox(cameraPos + cameraSpeed * cameraFront):
            cameraPos += cameraSpeed * cameraFront
        else:
            cameraPos -= cameraSpeed * cameraFront
    if key == glfw.KEY_S and (action == glfw.PRESS or action == glfw.REPEAT):
        if skybox(cameraPos - cameraSpeed * cameraFront):
            cameraPos -= cameraSpeed * cameraFront
        else:
            cameraPos += cameraSpeed * cameraFront
    if key == glfw.KEY_A and (action == glfw.PRESS or action == glfw.REPEAT):
        if skybox(cameraPos - glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed):
            cameraPos -= glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        else:
            cameraPos += glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
    if key == glfw.KEY_D and (action == glfw.PRESS or action == glfw.REPEAT):
        if skybox(cameraPos + glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed):
            cameraPos += glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        else:
            cameraPos -= glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
    if key == glfw.KEY_P and action == glfw.PRESS:
        wireframe = not wireframe

    if key == 61 and mods == 0 and (action == glfw.PRESS or action == glfw.REPEAT):
        cameraSpeed += 0.01
    if key == 45 and mods == 0 and (action == glfw.PRESS or action == glfw.REPEAT):
        cameraSpeed -= 0.01
    if key == 61 and mods == 1 and (action == glfw.PRESS or action == glfw.REPEAT):
        sensitivity += 0.01
    if key == 45 and mods == 1 and (action == glfw.PRESS or action == glfw.REPEAT):
        sensitivity -= 0.01

yaw = -90.0
pitch = 0.0
lastX = largura/2
lastY = altura/2

def mouse_event(window, xpos, ypos):
    global cameraFront, yaw, pitch, lastX, lastY, sensitivity

    xoffset = xpos - lastX
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos

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

def draw_skybox():
    angle = 0.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0;
    t_x = 0.0; t_y = 1000.0; t_z = 0.0;
    s_x = 1024; s_z = 1024; s_y = 1024;
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['skybox']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['skybox']['start'], modelos['skybox']['size'])

def draw_terrain():
    angle = 0.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0;
    t_x = 0.0; t_y = 0.0; t_z = 0.0;
    s_x = s_z = 1024; s_y = 1;
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['terrain']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['terrain']['start'], modelos['terrain']['size'])

def draw_road():
    angle = 90.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0;
    t_x = 600.0; t_y = 1.0; t_z = 0.0;
    s_x = 102.4; s_z = 1024; s_y = 1;
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['road']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['road']['start'], modelos['road']['size'])

def draw_house():
    angle = 0.0
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = 0.0; t_y = 0.0; t_z = 600.0
    s_x = s_y = s_z = 4;
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['house']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['house']['start'], modelos['house']['size'])

def draw_person():
    angle = 90.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = -680.0; t_y = 2.0; t_z = 000.0
    s_x = s_y = s_z = 0.65;
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['person']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['person']['start'], modelos['person']['size'])

def draw_uganda_knuckles():
    angle = 0.0;
    r_x = 0.0; r_y = 0.0; r_z = 1.0
    t_x = 90.0; t_y = 2.0; t_z = 610.0
    s_x = s_y = s_z = 8
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['uganda_knuckles']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['uganda_knuckles']['start'], modelos['uganda_knuckles']['size'])

def draw_statue():
    angle = 90.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = -610.0; t_y = 2.0; t_z = -90.0
    s_x = s_y = s_z = 0.35
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['statue']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['statue']['start'], modelos['statue']['size'])

def draw_tree_1():
    angle = 0.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = 50.0; t_y = 2.0; t_z = 20.0
    s_x = s_y = s_z = 100
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['tree']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['tree']['start'], modelos['tree']['size'])

def draw_tree_2():
    angle = 0.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = -20.0; t_y = 2.0; t_z = -850.0
    s_x = s_y = s_z = 85
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['tree']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['tree']['start'], modelos['tree']['size'])

def draw_tree_3():
    angle = 30.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = -600.0; t_y = 2.0; t_z = 0.0
    s_x = s_y = s_z = 90
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['tree']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['tree']['start'], modelos['tree']['size'])

def draw_deer(angle):
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = 0.0; t_y = 2.0; t_z = 0.0
    s_x = s_y = s_z = 90
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['deer']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['deer']['start'], modelos['deer']['size'])

def draw_bench():
    angle = 0
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = -330.0; t_y = 2.0; t_z = 350.0
    s_x = s_y = s_z = 50
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['bench']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['bench']['start'], modelos['bench']['size'])
    glBindTexture(GL_TEXTURE_2D, modelos['bench']['texture_id']+1)
    glDrawArrays(GL_TRIANGLES, modelos['bench']['start'], modelos['bench']['size'])

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
    #                                fov                aspect ratio    near  far
    mat_projection = glm.perspective(glm.radians(90.0), largura/altura, 0.01, 5000.0)
    mat_projection = np.array(mat_projection)
    return mat_projection

#}}}
#{{{ LOOP

glfw.show_window(window)
glfw.set_cursor_pos(window, largura/2, altura/2)

glEnable(GL_DEPTH_TEST)

wave_obj = sa.WaveObject.from_wave_file("media/floral.wav")
play_obj = wave_obj.play()

while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(0.0, 0.0, 0.0, 1.0)

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if wireframe else GL_FILL)

    draw_skybox()
    draw_terrain()
    draw_road()
    draw_house()
    draw_person()
    draw_uganda_knuckles()
    draw_statue()
    draw_tree_1()
    draw_tree_2()
    draw_tree_3()
    draw_bench()
    draw_deer(deer_angle)
    deer_angle += 0.5

    mat_view = view()
    loc_view = glGetUniformLocation(program, "view")
    glUniformMatrix4fv(loc_view, 1, GL_FALSE, mat_view)

    mat_projection = projection()
    loc_projection = glGetUniformLocation(program, "projection")
    glUniformMatrix4fv(loc_projection, 1, GL_FALSE, mat_projection)

    glfw.swap_buffers(window)

glfw.terminate()
#}}}

