#version 330 core

in vec2 uv;
out vec4 fragColor;

void main() {
    gl_Position = vec4(uv, 0.0, 1.0);
}
