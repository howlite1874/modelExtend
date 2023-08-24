#version 450
layout(location = 0) in vec2 texCoords;

layout(set = 1,binding = 0) uniform sampler2D aoMap;  

layout(location = 0) out vec4 oColor;

void main()
{
    const float alphaThreshold = 0.5;
    vec4 texColor = texture(aoMap, texCoords);

    // Discard the fragment if its alpha value is below the threshold
    if (texColor.a < alphaThreshold) {
        discard;
    }

    //oColor = texColor;
}