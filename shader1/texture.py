import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from PIL import Image

def read_shader_file(filename):
    with open(filename, 'r') as file:
        return file.read()

# Function to compile shaders
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    # Check compilation status
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    
    return shader

# def use(texture):
#     glActiveTexture(GL_TEXTURE0)
#     glBindTexture(GL_TEXTURE_2D, texture)

# def destory(texture):
#     glDeleteTextures(1, (texture))

# Initialize Pygame and OpenGL
pg.init()
display = (1024, 1024)
# display = (2048, 2048)
pg.display.set_mode(display, DOUBLEBUF | OPENGL)

# Load shader source code from files
vertex_shader_source = read_shader_file('vertex_shader.glsl')
fragment_shader_source = read_shader_file('shader1/frag2texture.glsl')  # Modified fragment shader

# Compile vertex and fragment shaders
vertex_shader = compile_shader(vertex_shader_source, GL_VERTEX_SHADER)
fragment_shader = compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER)

# Create and link shader program
shader_program = glCreateProgram()
glAttachShader(shader_program, vertex_shader)
glAttachShader(shader_program, fragment_shader)
glLinkProgram(shader_program)

# Use the shader program
glUseProgram(shader_program)

# Set up uniforms (you should set these based on your specific camera setup)
camera_position_uniform = glGetUniformLocation(shader_program, "cameraPosition")
glUniform3f(camera_position_uniform, 0.0, 0.0, 5.0)

resolution_uniform = glGetUniformLocation(shader_program, "iResolution")
glUniform2f(resolution_uniform, display[0], display[1])

# Load texture
# image = Image.open('probe1.png')
# width, height = image.size
# image_data = image.tobytes()
image = pg.image.load('probe_data/probe1.png').convert()
width, height = image.get_rect().size
print(f'width height are {width, height}')
image_data = pg.image.tostring(image, "RGBA")

texture_id = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture_id)
# glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data
# glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
# glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
# glGenerateMipmap(GL_TEXTURE_2D)


# Use the texture in the shader as LightFieldSurface
lightfield_surface_uniform = glGetUniformLocation(shader_program, "lightFieldSurface.radianceProbeGrid")
glUniform1i(lightfield_surface_uniform, 0)  # 0 corresponds to GL_TEXTURE0

# Main loop
while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            quit()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Draw quad with texture
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(-1, -1)
    glTexCoord2f(1, 0); glVertex2f(1, -1)
    glTexCoord2f(1, 1); glVertex2f(1, 1)
    glTexCoord2f(0, 1); glVertex2f(-1, 1)
    glEnd()

    pg.display.flip()
    pg.time.wait(10)
