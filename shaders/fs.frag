#version 450

layout (set = 1, binding = 0) uniform sampler2D tex_sampler;

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec4 inColor;

layout (location = 0) out vec4 fragColor;

void main() {
    fragColor = inColor * texture(tex_sampler, inUV);
}
