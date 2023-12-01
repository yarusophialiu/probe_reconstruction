import pygame
import math
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from PIL import Image
# import glfw

# quadVAO = 0  # Vertex Array Object ID
# quadVBO = 0  # Vertex Buffer Object ID

def init_quad():
    global quadVAO, quadVBO

    # Define the vertices of a quad
    vertices = np.array([
        -1,  1, 0.0,  # Top-left vertex
         1,  1, 0.0,  # Top-right vertex
         1, -1, 0.0,  # Bottom-right vertex
        -1, -1, 0.0   # Bottom-left vertex
    ], dtype=np.float32)

    quadVAO = glGenVertexArrays(1)  
    glBindVertexArray(quadVAO)

    # Generate and bind a VBO
    quadVBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO)

    # Fill the buffer with vertex data
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # # Unbind the buffer
    # glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Set up vertex attribute pointers
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), None)
    # glEnableVertexAttribArray(0)

    # Unbind the VAO and VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)


def render_quad():
    global quadVBO
    global quadVAO
    

    # Bind the VBO
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO)
    glBindVertexArray(quadVAO)

    # Enable and specify the vertex attribute pointers
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, None)

    # Draw the quad
    glDrawArrays(GL_QUADS, 0, 4)

    # Cleanup
    # glDisableClientState(GL_VERTEX_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)



def read_shader_file(filename):
    with open(filename, 'r') as file:
        return file.read()
    
    
def normalize(v, eps=1e-8):
    dist = np.linalg.norm(v)
    return v / (dist + eps)


def lookAt(eye, at, up):
    """Viewing transformation."""
    z = normalize(eye - at)
    x = normalize(np.cross(up, z))
    y = normalize(np.cross(z, x))
    A = np.column_stack((x, y, z, eye))
    A = np.row_stack((A, np.array([0, 0, 0, 1])))
    return np.linalg.inv(A)

class Camera:
    """
    at: position where the camera looks at
    up: up direction of the camera
    """
    def __init__(self, eye, at, up, fov=None):
        self.eye = eye
        self.at = at
        self.up = up
        self.fov = fov
    
def get_params(cam, width, height):
    aspect_ratio = width / height
    world_to_camera = lookAt(cam.eye, cam.at, cam.up)
    camera_to_world = np.linalg.inv(world_to_camera)

    fov = cam.fov
    radians = fov * math.pi / 180
    h_c = 2 * np.tan(radians/2)
    w_c = aspect_ratio * h_c

    a, b = -w_c/2, h_c/2
    dx, dy = w_c/width, h_c/height

    # print(f"h_c {h_c}\n")
    # print(f"width, height {width, height}\n")
    return camera_to_world, dx, dy, a, b


# Function to compile shaders
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    # Check compilation status
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    
    return shader

def create_shader_program(vertex_shader_source, fragment_shader_source):
    vertex_shader = compile_shader(vertex_shader_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER)

    # Create and link shader program
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)

    # Check linking status
    if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(shader_program))

    return shader_program

def load_texture(image_path):
    image = Image.open(image_path)
    width, height = image.size
    image_data = image.tobytes()

    # Generate and bind texture
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D_ARRAY, texture_id)

    # Allocate storage for the texture array
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGB8, width, height, 2)

    # Upload texture data to the texture array
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, width, height, 1, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
    # Add another layer if you have a second image
    # glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 1, width, height, 1, GL_RGB, GL_UNSIGNED_BYTE, second_image_data)

    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)


    return texture_id


# Initialize Pygame and OpenGL
pygame.init()
width, height = 1024, 1024
# width, height = 2048, 2048
display = (width, height)

pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

# Load shader source code from files
vertex_shader_source = read_shader_file('vertex_shader.glsl')
fragment_shader_source = read_shader_file('fragment_shader.glsl')

# Compile vertex and fragment shaders
vertex_shader = compile_shader(vertex_shader_source, GL_VERTEX_SHADER)
fragment_shader = compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER)

# Create shader program
shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
glUseProgram(shader_program)

# camera position
camX, camY, camZ = -0.5, 4.0, 0.0
cameraPosition = np.array([camX, camY, camZ])
atX, atY, atZ = 0.956, 3.591, 3.563
at = np.array([atX, atY, atZ])

# Set up uniforms (you should set these based on your specific camera setup)
camera_position_uniform = glGetUniformLocation(shader_program, "cameraPosition")
look_at_uniform = glGetUniformLocation(shader_program, "lookAt")
# glUniform3f(camera_position_uniform, 0.0, 0.0, 5.0) # world space
# glUniform3f(look_at_uniform, 0.0, 0.0, -1.0) # world space
glUniform3f(camera_position_uniform, camX, camY, camZ) # world space
glUniform3f(look_at_uniform, atX, atY, atZ) # world space

resolution_uniform = glGetUniformLocation(shader_program, "iResolution")
glUniform2f(resolution_uniform, display[0], display[1]) # set width and height


# Load textures
IMAGE_FOLDER = "probe_data"
image_paths = [f'{IMAGE_FOLDER}/probe1.png', f'{IMAGE_FOLDER}/probe2.png']
num_images = len(image_paths)

# Create a 2D texture array
texture_array_id = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D_ARRAY, texture_array_id)
# TODO: change to ideal size
textureWidth, textureHeight = 2048, 2048

# Specify the storage for the texture array
# allocate storage for a three-dimensional texture, 
# and in this case, it's specifically used for a 2D texture array 
glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGB8, textureWidth, textureHeight, num_images)

# Load each image into a different layer of the texture array
for layer, image_path in enumerate(image_paths[:1]):
    # image data is stored with the origin at the top-left corner
    image = Image.open(image_path)
    # print(image.size)
    image_data = image.tobytes()
    # update a portion of a three-dimensional texture, 2D texture array here
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer, textureWidth, textureHeight, 1, GL_RGB, GL_UNSIGNED_BYTE, image_data)

# Use the texture array in the shader as LightFieldSurface
# retrieves the location of a uniform variable in the shader program
lightfield_surface_uniform = glGetUniformLocation(shader_program, "L1.radianceProbeGrid")
# sets the value of the specified integer uniform variable
glUniform1i(lightfield_surface_uniform, 0)  # 0 corresponds to GL_TEXTURE0, first texture unit
probe1_position_uniform = glGetUniformLocation(shader_program, "L1.probePosition")
glUniform3f(probe1_position_uniform, -0.5, 4.0, 0.0) 


# pass matrix and params to frag glsl fo ray direction computation
up = np.array([0, 1, 0])
fov = 70
cam = Camera(cameraPosition, at, up, fov)
camera_to_world, dx, dy, a, b = get_params(cam, width, height)
# print(f'camera_to_world \n {camera_to_world}')

matrix_location = glGetUniformLocation(shader_program, "cameraToWorld")
glUniformMatrix4fv(matrix_location, 1, GL_FALSE, camera_to_world)

dXY = glGetUniformLocation(shader_program, "dxy")
glUniform2fv(dXY, 1, np.array([dx, dy]))

ab = glGetUniformLocation(shader_program, "ab")
glUniform2fv(ab, 1, np.array([a, b]))


init_quad()
# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Draw quad with texture
    # glBegin(GL_QUADS)

    global quadVBO

    # Initialize GLFW and create a window...

    # Initialize GLEW...

    # Initialize the quad


    # Clear the screen...

    # Render the quad
    render_quad()
    # Cleanup code...


    # glTexCoord2f(0, 0); glVertex2f(-1, -1)
    # glTexCoord2f(1, 0); glVertex2f(1, -1)
    # glTexCoord2f(1, 1); glVertex2f(1, 1)
    # glTexCoord2f(0, 1); glVertex2f(-1, 1)
    # glEnd()

    pygame.display.flip()
    pygame.time.wait(10)
