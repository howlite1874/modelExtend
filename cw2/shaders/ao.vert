#version 450
#extension GL_EXT_debug_printf : enable
layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec2 iTexcoord;

//std140
layout(set = 0,binding = 0) uniform UScene{
	mat4 camera;
	mat4 projection;
	mat4 projCamera;
}uScene;


layout(location = 0) out vec2 texCoords;


void main()
{
	texCoords = iTexcoord;
	gl_Position = uScene.projCamera * vec4(iPosition,1.f);

}

