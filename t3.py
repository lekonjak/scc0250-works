#!/usr/bin/env python
# coding: utf-8

# Ricardo Alves de Araujo - 9364890
# Tiago Esperança Triques - 9037713

# Use WASD and mouse to move and look around
# U and P to increase or decrease the external light source intensity
# L to toggle internal light source
# O to toggle wireframe view mode
# Q to quit

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
window = glfw.create_window(largura, altura, "T3", None, None)
glfw.make_context_current(window)
glfw.set_window_pos(window, int(1366/4), 0)

#}}}
#{{{GLSL

vertex_code = """
        attribute vec3 position;
        attribute vec2 texture_coord;
        attribute vec3 normals;

        varying vec2 out_texture;
        varying vec3 out_fragPos;
        varying vec3 out_normal;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main(){
            gl_Position = projection * view * model * vec4(position,1.0);
            out_texture = vec2(texture_coord);
            out_fragPos = vec3(model * vec4(position, 1.0));
            out_normal = normals;
        }
        """

fragment_code = """
        #version 120

        uniform vec3 lightPos1;
        uniform vec3 lightPos2;
        uniform vec3 viewPos;

        uniform float ka;
        uniform float kd;
        uniform float ks;
        uniform float ns;

        // parametro com a cor da(s) fonte(s) de iluminacao
        uniform vec3 lightColor = vec3(201.0/255.0, 226.0/255.0, 255.0/255.0);
        uniform vec3 lightColor2;

        varying vec2 out_texture;
        varying vec3 out_normal;
        varying vec3 out_fragPos;
        uniform sampler2D samplerTexture;

        void main(){
            vec3 ambient = ka * lightColor;
            // Luz #1
            vec3 norm1 = normalize(out_normal);
            vec3 lightDir1 = normalize(lightPos1 - out_fragPos);
            float diff1 = max(dot(norm1, lightDir1), 0.0);
            vec3 diffuse1 = kd * diff1 * lightColor;

            vec3 viewDir1 = normalize(viewPos - out_fragPos);
            vec3 reflectDir1 = reflect(-lightDir1, norm1);
            float spec1 = pow(max(dot(viewDir1, reflectDir1), 0.0), ns);
            vec3 specular1 = ks * spec1 * lightColor;

            // Luz #2 (FIXA)
            vec3 norm2 = normalize(out_normal);
            vec3 lightDir2 = normalize(lightPos2 - out_fragPos);
            float diff2 = max(dot(norm2, lightDir2), 0.0);
            vec3 diffuse2 = kd * diff2 * lightColor2;

            vec3 viewDir2 = normalize(viewPos - out_fragPos);
            vec3 reflectDir2 = reflect(-lightDir2, norm2);
            float spec2 = pow(max(dot(viewDir2, reflectDir2), 0.0), ns);
            vec3 specular2 = ks * spec2 * lightColor2;

            //Attenuation
	    float distance = length(lightPos2 - out_fragPos);
            float attenuation = 1.0f;
            if(distance < 300.0f) {
                attenuation /= 0.001f * distance;
            }

	    //Final light
	    vec3 diffuse  = diffuse1  + (diffuse2  * attenuation);
	    vec3 specular = specular1 + (specular2 * attenuation);

            // Combinando as duas fontes
            vec4 texture = texture2D(samplerTexture, out_texture);
            vec4 result = vec4((ambient + diffuse + specular), 1.0) * texture;
            gl_FragColor = result;
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
    normals = []
    texture_coords = []
    faces = []

    material = None

    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':
            vertices.append(values[1:4])
        if values[0] == 'vn':
            normals.append(values[1:4])
        elif values[0] == 'vt':
            texture_coords.append(values[1:3])
        elif values[0] in ('usemtl', 'usemat'):
            material = values[1]
        elif values[0] == 'f':
            face = []
            face_texture = []
            face_normals = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0]))
                face_normals.append(int(w[2]))
                if len(w) >= 2 and len(w[1]) > 0:
                    face_texture.append(int(w[1]))
                else:
                    face_texture.append(0)

            faces.append((face, face_texture, face_normals, material))

    model = {}
    model['vertices'] = vertices
    model['texture'] = texture_coords
    model['faces'] = faces
    model['normals'] = normals

    return model

glEnable(GL_TEXTURE_2D)
qtd_texturas = 10
textures = glGenTextures(qtd_texturas)

def load_texture_from_file(texture_id, img_textura):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    img = Image.open(img_textura)
    img_width = img.size[0]
    img_height = img.size[1]
    image_data = img.convert("RGBA").tobytes("raw", "RGBA", 0, -1)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_width, img_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)

#}}}
#{{{ RANDOM GLOBAL VARIABLES

texture_count = 0
modelos = {}

cameraSpeed = 5
sensitivity = 0.15

shouldIncrement = True
lightOn = False
lpx = 0.0
lpy = 0.0
lpz = 0.0

intensity = 1.0

#loc_light_pos = glGetUniformLocation(program, "lightPos")
#glUniform3f(loc_light_pos, 0, 100, 600)

#}}}
#{{{ LOAD MODEL AND TEXTURES

# Enable textures, allocate memory
glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
glEnable(GL_BLEND)
glDisable(GL_CULL_FACE)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glEnable(GL_LINE_SMOOTH)

glEnable(GL_TEXTURE_2D)
qtd_texturas = 100
textures = glGenTextures(qtd_texturas)

vertices_list = []
normals_list = []
textures_coord_list = []

# Skybox
modelo = load_model_from_file('models/skybox/skybox.obj')
modelos['skybox'] = {}
modelos['skybox']['n_texturas'] = 1
modelos['skybox']['start'] = len(vertices_list)
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id-1] )
modelos['skybox']['end'] = len(vertices_list)
modelos['skybox']['size'] = modelos['skybox']['end'] - modelos['skybox']['start']
modelos['skybox']['texture_id'] = texture_count
load_texture_from_file(modelos['skybox']['texture_id'], 'models/skybox/texture.png')
texture_count += 1
print(f"Quantidade de vértices de skybox.obj {modelos['skybox']['size']}")

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
    for normal_id in face[2]:
        normals_list.append(modelo['normals'][normal_id-1])
modelos['terrain']['end'] = len(vertices_list)
modelos['terrain']['size'] = modelos['terrain']['end'] - modelos['terrain']['start']
modelos['terrain']['texture_id'] = texture_count
load_texture_from_file(modelos['terrain']['texture_id'], 'models/terrain/grass.jpg')
texture_count += 1
print(f"Quantidade de vértices de terrain.obj {modelos['terrain']['size']}")

# Rua
modelo = load_model_from_file('models/terrain/road.obj')
modelos['road'] = {}
modelos['road']['n_texturas'] = 1
modelos['road']['start'] = len(vertices_list)
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id-1] )
modelos['road']['end'] = len(vertices_list)
modelos['road']['size'] = modelos['road']['end'] - modelos['road']['start']
modelos['road']['texture_id'] = texture_count
load_texture_from_file(modelos['road']['texture_id'], 'models/terrain/road.jpg')
texture_count += 1
print(f"Quantidade de vértices de road.obj {modelos['road']['size']}")

# Casa
modelo = load_model_from_file('models/house/house2.obj')
modelos['house'] = {}
modelos['house']['n_texturas'] = 1
modelos['house']['start'] = len(vertices_list)
print('Processando modelo house.obj')
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id-1] )
modelos['house']['end'] = len(vertices_list)
modelos['house']['size'] = modelos['house']['end'] - modelos['house']['start']
modelos['house']['texture_id'] = texture_count
load_texture_from_file(modelos['house']['texture_id'], 'models/house/texture.jpg')
texture_count += 1
print(f"Quantidade de vértices de house.obj {modelos['house']['size']}")

# Denis
modelo = load_model_from_file('models/person/untitled.obj')
modelos['person'] = {}
modelos['person']['n_texturas'] = 1
modelos['person']['start'] = len(vertices_list)
print('Processando modelo person.obj')
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
    for normal_id in face[2]:
        normals_list.append(modelo['normals'][normal_id-1])
modelos['person']['end'] = len(vertices_list)
modelos['person']['size'] = modelos['person']['end'] - modelos['person']['start']
modelos['person']['texture_id'] = texture_count
load_texture_from_file(modelos['person']['texture_id'], 'models/person/denis.jpg')
texture_count += 1
print(f"Quantidade de vértices de person.obj {modelos['person']['size']}")

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
    for normal_id in face[2]:
        normals_list.append(modelo['normals'][normal_id-1])
modelos['uganda_knuckles']['end'] = len(vertices_list)
modelos['uganda_knuckles']['size'] = modelos['uganda_knuckles']['end'] - modelos['uganda_knuckles']['start']
modelos['uganda_knuckles']['texture_id'] = texture_count
load_texture_from_file(modelos['uganda_knuckles']['texture_id'], 'models/uganda/Knuckles_Texture.jpg')
texture_count += 1
print(f"Quantidade de vértices de uganda_knuckles.obj {modelos['uganda_knuckles']['size']}")

# Statue
modelo = load_model_from_file('models/statue/untitled.obj')
modelos['statue'] = {}
modelos['statue']['n_texturas'] = 1
modelos['statue']['start'] = len(vertices_list)
print('Processando modelo statue.obj')
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id-1] )
modelos['statue']['end'] = len(vertices_list)
modelos['statue']['size'] = modelos['statue']['end'] - modelos['statue']['start']
modelos['statue']['texture_id'] = texture_count
load_texture_from_file(modelos['statue']['texture_id'], 'models/statue/texture.jpg')
texture_count += 1
print(f"Quantidade de vértices de statue.obj {modelos['statue']['size']}")

# Tree
modelo = load_model_from_file('models/tree/tree.obj')
modelos['tree'] = {}
modelos['tree']['n_texturas'] = 4
modelos['tree']['start'] = len(vertices_list)
print('Processando modelo tree.obj')
faces_visited = []
for face in modelo['faces']:
    if face[3] not in faces_visited:
        modelos['tree'][f'{face[3]}'] = len(vertices_list)
        faces_visited.append(face[3])
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
    for normal_id in face[2]:
        normals_list.append(modelo['normals'][normal_id-1])
modelos['tree']['end'] = len(vertices_list)
modelos['tree']['size'] = modelos['tree']['end'] - modelos['tree']['start']
modelos['tree']['texture_id'] = texture_count
load_texture_from_file(modelos['tree']['texture_id'], 'models/tree/as12brk1.tif')
load_texture_from_file(modelos['tree']['texture_id']+1, 'models/tree/as12brn1.tif')
load_texture_from_file(modelos['tree']['texture_id']+2, 'models/tree/as12lef1.tif')
load_texture_from_file(modelos['tree']['texture_id']+3, 'models/tree/as12lef2.tif')
texture_count += modelos['tree']['n_texturas']
print(f"Quantidade de vértices de tree.obj {modelos['tree']['size']}")

# Deer
modelo = load_model_from_file('models/deer/deer.obj')
modelos['deer'] = {}
modelos['deer']['n_texturas'] = 1
modelos['deer']['start'] = len(vertices_list)
print('Processando modelo deer.obj')
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id-1] )
modelos['deer']['end'] = len(vertices_list)
modelos['deer']['size'] = modelos['deer']['end'] - modelos['deer']['start']
modelos['deer']['texture_id'] = texture_count
load_texture_from_file(modelos['deer']['texture_id'], 'models/deer/texture.jpg')
texture_count += 1
print(f"Quantidade de vértices de deer.obj {modelos['deer']['size']}")

# Bench
modelo = load_model_from_file('models/bench/bench.obj')
modelos['bench'] = {}
modelos['bench']['n_texturas'] = 2
modelos['bench']['start'] = len(vertices_list)
print('Processando modelo bench.obj')
faces_visited = []
for face in modelo['faces']:
    if face[3] not in faces_visited:
        modelos['bench'][f'{face[3]}'] = len(vertices_list)
        faces_visited.append(face[3])
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id-1] )
modelos['bench']['end'] = len(vertices_list)
modelos['bench']['size'] = len(vertices_list) - modelos['bench']['start']
modelos['bench']['texture_id'] = texture_count
load_texture_from_file(modelos['bench']['texture_id'], 'models/bench/texture_wood.jpg')
load_texture_from_file(modelos['bench']['texture_id']+1, 'models/bench/texture_steel.jpg')
texture_count += modelos['bench']['n_texturas']
print(f"Quantidade de vértices de bench.obj {modelos['bench']['size']}")

# Bus
modelo = load_model_from_file('models/bus/bus.obj')
modelos['bus'] = {}
modelos['bus']['n_texturas'] = 3
modelos['bus']['start'] = len(vertices_list)
print('Processando modelo bus.obj')
faces_visited = []
for face in modelo['faces']:
    if face[3] not in faces_visited:
        modelos['bus'][f'{face[3]}'] = len(vertices_list)
        faces_visited.append(face[3])
    for vertice_id in face[0]:
        vertices_list.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append(modelo['texture'][texture_id-1])
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id-1] )
modelos['bus']['end'] = len(vertices_list)
modelos['bus']['size'] = modelos['bus']['end'] - modelos['bus']['start']
modelos['bus']['texture_id'] = texture_count
load_texture_from_file(modelos['bus']['texture_id'],   'models/bus/Textures/wheel_d.jpg')
load_texture_from_file(modelos['bus']['texture_id']+1, 'models/bus/Textures/inside_d.jpg')
load_texture_from_file(modelos['bus']['texture_id']+2, 'models/bus/Textures/corpus_d.jpg')
texture_count += modelos['bus']['n_texturas']
print(f"Quantidade de vértices de bus.obj {modelos['bus']['size']}")

print(modelos)

# Request a buffer slot from GPU
buffer = glGenBuffers(3)

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

textures = np.zeros(len(textures_coord_list), [("position", np.float32, 2)])
textures['position'] = textures_coord_list

# Upload data
glBindBuffer(GL_ARRAY_BUFFER, buffer[1])
glBufferData(GL_ARRAY_BUFFER, textures.nbytes, textures, GL_STATIC_DRAW)
stride = textures.strides[0]
offset = ctypes.c_void_p(0)
loc_texture_coord = glGetAttribLocation(program, "texture_coord")
glEnableVertexAttribArray(loc_texture_coord)
glVertexAttribPointer(loc_texture_coord, 2, GL_FLOAT, False, stride, offset)

normals = np.zeros(len(normals_list), [("position", np.float32, 3)])
normals['position'] = normals_list

# Upload coordenadas normals de cada vertice
glBindBuffer(GL_ARRAY_BUFFER, buffer[2])
glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
stride = normals.strides[0]
offset = ctypes.c_void_p(0)
loc_normals_coord = glGetAttribLocation(program, "normals")
glEnableVertexAttribArray(loc_normals_coord)
glVertexAttribPointer(loc_normals_coord, 3, GL_FLOAT, False, stride, offset)

#}}}
#{{{ INPUT EVENTS

cameraPos   = glm.vec3(0.0, 100, 600);
cameraFront = glm.vec3(0.0, 0.0, 0.0);
cameraUp    = glm.vec3(0.0, 1.0, 0.0);

def skybox(pos):
    if -1024 < pos[0] < 1024 and 10 < pos[1] < 1024 and -1024 < pos[2] < 1024:
        return True
    return False

wireframe = False

def key_event(window,key,scancode,action,mods):
    global cameraPos, cameraFront, cameraUp
    global wireframe, scale, cameraSpeed, sensitivity
    global lightOn, intensity

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

    if key == glfw.KEY_O and action == glfw.PRESS:
        wireframe = not wireframe

    if key == glfw.KEY_L and action == glfw.PRESS:
        loc_light = glGetUniformLocation(program, "lightColor2")
        if lightOn:
            glUniform3f(loc_light, 0.0, 0.0, 0.0)
        else:
            glUniform3f(loc_light, 255.0/255.0, 147.0/255.0, 41.0/255.0)
        lightOn = not lightOn

    if key == glfw.KEY_U and action == glfw.PRESS:
        if intensity + 0.05 < 1.0:
            intensity += 0.05
        loc_light = glGetUniformLocation(program, "lightColor")
        glUniform3f(loc_light, (201.0/255.0) * intensity, (226.0/255.0) * intensity, (255.0/255.0) * intensity )
    if key == glfw.KEY_P and action == glfw.PRESS:
        if intensity - 0.05 > 0.0:
            intensity -= 0.05
        loc_light = glGetUniformLocation(program, "lightColor")
        glUniform3f(loc_light, (201.0/255.0) * intensity, (226.0/255.0) * intensity, (255.0/255.0) * intensity )

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
    t_x = 0.0; t_y = 0.0; t_z = 0.0;
    s_x = 1024; s_z = 1024; s_y = 1024;
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    ka = 1.0
    kd = 0.4
    ks = 0.0
    ns = 1
    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)
    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)
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
    ka = 1.0
    kd = 0.1
    ks = 0.0
    ns = 1
    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)
    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)
    glBindTexture(GL_TEXTURE_2D, modelos['terrain']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['terrain']['start'], modelos['terrain']['size'])

def draw_road():
    for i in range(-12,13):
        angle = 90.0;
        r_x = 0.0; r_y = 1.0; r_z = 0.0;
        t_x = 600.0; t_y = 1.0; t_z = i*100.0;
        s_x = 100; s_z = 100; s_y = 1;
        mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
        loc_model = glGetUniformLocation(program, "model")
        ka = 0.7
        kd = 0.2
        ks = 0.0
        ns = 1
        loc_ka = glGetUniformLocation(program, "ka")
        glUniform1f(loc_ka, ka)
        loc_kd = glGetUniformLocation(program, "kd")
        glUniform1f(loc_kd, kd)
        loc_ks = glGetUniformLocation(program, "ks")
        glUniform1f(loc_ks, ks)
        loc_ns = glGetUniformLocation(program, "ns")
        glUniform1f(loc_ns, ns)
        glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
        glBindTexture(GL_TEXTURE_2D, modelos['road']['texture_id'])
        glDrawArrays(GL_TRIANGLES, modelos['road']['start'], modelos['road']['size'])

def draw_house():
    angle = 0.0
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = 0.0; t_y = 0.0; t_z = 600.0
    s_x = s_y = s_z = 5;
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    ka = 0.5
    kd = 0.2
    ks = 0.0
    ns = 1
    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)
    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['house']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['house']['start'], modelos['house']['size'])

def draw_person():
    angle = 90.0;
    angle = 0.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = -680.0; t_y = 2.0; t_z = 0.0
    t_x = 0.0; t_y = 2.0; t_z = 680.0
    s_x = s_y = s_z = 0.65;
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    ka = 0.5
    kd = 0.2
    ks = 0.0
    ns = 1
    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)
    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)
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
    ka = 0.5
    kd = 0.3
    ks = 1.0
    ns = 2048

    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)
    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)
    glBindTexture(GL_TEXTURE_2D, modelos['uganda_knuckles']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['uganda_knuckles']['start'], modelos['uganda_knuckles']['size'])

def draw_statue():
    angle = 0.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = -90.0; t_y = 2.0; t_z = 610.0
    s_x = s_y = s_z = 0.35
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    ka = 0.5
    kd = 0.2
    ks = 1.0
    ns = 1024
    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)
    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['statue']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['statue']['start'], modelos['statue']['size'])

def draw_tree_1():
    angle = 0.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = 50.0; t_y = 2.0; t_z = 50.0
    s_x = s_y = s_z = 100
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    ka = 0.9
    kd = 0.6
    ks = 0.0
    ns = 1
    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)
    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['tree']['texture_id']+3)
    glDrawArrays(GL_TRIANGLES, modelos['tree']['AS12_Leaf2'], modelos['tree']['AS12_Leaf1']-modelos['tree']['AS12_Leaf2'])
    glBindTexture(GL_TEXTURE_2D, modelos['tree']['texture_id']+2)
    glDrawArrays(GL_TRIANGLES, modelos['tree']['AS12_Leaf1'], modelos['tree']['AS12_Bark1']-modelos['tree']['AS12_Leaf1'])
    glBindTexture(GL_TEXTURE_2D, modelos['tree']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['tree']['AS12_Bark1'], modelos['tree']['AS12_Branch1']-modelos['tree']['AS12_Bark1'])
    glBindTexture(GL_TEXTURE_2D, modelos['tree']['texture_id']+1)
    glDrawArrays(GL_TRIANGLES, modelos['tree']['AS12_Branch1'], modelos['tree']['end']-modelos['tree']['AS12_Branch1'])

def draw_tree_2():
    angle = 0.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = -20.0; t_y = 2.0; t_z = -850.0
    s_x = s_y = s_z = 85
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    ka = 0.9
    kd = 0.5
    ks = 1.0
    ns = 1
    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)
    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)
    glBindTexture(GL_TEXTURE_2D, modelos['tree']['texture_id']+3)
    glDrawArrays(GL_TRIANGLES, modelos['tree']['AS12_Leaf2'], modelos['tree']['AS12_Leaf1']-modelos['tree']['AS12_Leaf2'])
    glBindTexture(GL_TEXTURE_2D, modelos['tree']['texture_id']+2)
    glDrawArrays(GL_TRIANGLES, modelos['tree']['AS12_Leaf1'], modelos['tree']['AS12_Bark1']-modelos['tree']['AS12_Leaf1'])
    glBindTexture(GL_TEXTURE_2D, modelos['tree']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['tree']['AS12_Bark1'], modelos['tree']['AS12_Branch1']-modelos['tree']['AS12_Bark1'])
    glBindTexture(GL_TEXTURE_2D, modelos['tree']['texture_id']+1)
    glDrawArrays(GL_TRIANGLES, modelos['tree']['AS12_Branch1'], modelos['tree']['end']-modelos['tree']['AS12_Branch1'])

def draw_tree_3():
    angle = 0.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = -600.0; t_y = 2.0; t_z = 0.0
    s_x = s_y = s_z = 90
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    ka = 0.9
    kd = 0.4
    ks = 0.0
    ns = 1
    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)
    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['tree']['texture_id']+3)
    glDrawArrays(GL_TRIANGLES, modelos['tree']['AS12_Leaf2'], modelos['tree']['AS12_Leaf1']-modelos['tree']['AS12_Leaf2'])
    glBindTexture(GL_TEXTURE_2D, modelos['tree']['texture_id']+2)
    glDrawArrays(GL_TRIANGLES, modelos['tree']['AS12_Leaf1'], modelos['tree']['AS12_Bark1']-modelos['tree']['AS12_Leaf1'])
    glBindTexture(GL_TEXTURE_2D, modelos['tree']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['tree']['AS12_Bark1'], modelos['tree']['AS12_Branch1']-modelos['tree']['AS12_Bark1'])
    glBindTexture(GL_TEXTURE_2D, modelos['tree']['texture_id']+1)
    glDrawArrays(GL_TRIANGLES, modelos['tree']['AS12_Branch1'], modelos['tree']['end']-modelos['tree']['AS12_Branch1'])

def draw_deer():
    angle = 0
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = 0.0; t_y = 0.0; t_z = 100.0
    s_x = s_y = s_z = 90
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    ka = 1.0
    kd = 0.0
    ks = 0.0
    ns = 1
    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)
    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)
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
    ka = 1.0
    kd = 0.2
    ks = 0.0
    ns = 1
    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)
    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)
    glBindTexture(GL_TEXTURE_2D, modelos['bench']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['bench']['start'], modelos['bench']['Steel']-modelos['bench']['woods'])
    glBindTexture(GL_TEXTURE_2D, modelos['bench']['texture_id']+1)
    glDrawArrays(GL_TRIANGLES, modelos['bench']['Steel'], modelos['bench']['end']-modelos['bench']['Steel'])

def draw_bus(bus_z_pos):
    angle = -90
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    t_x = -600.0; t_y = 2.0; t_z = bus_z_pos
    s_x = s_y = s_z = 0.3
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    ka = 0.7
    kd = 0.7
    ks = 0.2
    ns = 256
    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)
    loc_ks = glGetUniformLocation(program, "ks")
    glUniform1f(loc_ks, ks)
    loc_ns = glGetUniformLocation(program, "ns")
    glUniform1f(loc_ns, ns)
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    glBindTexture(GL_TEXTURE_2D, modelos['bus']['texture_id'])
    glDrawArrays(GL_TRIANGLES, modelos['bus']['Wheel'], modelos['bus']['Glass']-modelos['bus']['Wheel'])
    glBindTexture(GL_TEXTURE_2D, modelos['bus']['texture_id']+1)
    glDrawArrays(GL_TRIANGLES, modelos['bus']['Inside'], modelos['bus']['Wheel']-modelos['bus']['Inside'])
    glBindTexture(GL_TEXTURE_2D, modelos['bus']['texture_id']+2)
    glDrawArrays(GL_TRIANGLES, modelos['bus']['Corpus'], modelos['bus']['Inside']-modelos['bus']['Corpus'])
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
    mat_projection = glm.perspective(glm.radians(90.0), largura/altura, 1, 5000.0)
    mat_projection = np.array(mat_projection)
    return mat_projection

#}}}
#{{{ LOOP

glfw.show_window(window)
glfw.set_cursor_pos(window, largura/2, altura/2)

glEnable(GL_DEPTH_TEST)

#wave_obj = sa.WaveObject.from_wave_file("media/floral.wav")
#play_obj = wave_obj.play()

last = glfw.get_time()
nbframes = 0

bus_z_pos = -1300
sunPos = 0

loc_light_pos = glGetUniformLocation(program, "lightPos2")
glUniform3f(loc_light_pos, 0.0, 50.0, 510.0)

while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(0.0, 0.0, 0.0, 0.0)

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
    draw_deer()
    draw_bus(bus_z_pos)
    bus_z_pos += 5

    if bus_z_pos > 1300:
        bus_z_pos = -1300

    sunPos += 0.05

    loc_light_pos = glGetUniformLocation(program, "lightPos1")
    glUniform3f(loc_light_pos, 1000*math.sin(sunPos), 500, 1000*math.cos(sunPos))

    mat_view = view()
    loc_view = glGetUniformLocation(program, "view")
    glUniformMatrix4fv(loc_view, 1, GL_FALSE, mat_view)

    mat_projection = projection()
    loc_projection = glGetUniformLocation(program, "projection")
    glUniformMatrix4fv(loc_projection, 1, GL_FALSE, mat_projection)

    loc_view_pos = glGetUniformLocation(program, "viewPos")
    glUniform3f(loc_view_pos, cameraPos[0], cameraPos[1], cameraPos[2])
    glfw.swap_buffers(window)


glfw.terminate()

#}}}

