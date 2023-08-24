#version 450

//specify the precision of floating point
precision highp float;

layout(location = 0) in vec2 texCoords;
layout(location = 1) in vec3 worldPos;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangents;
layout(location = 4) in mat3 tbn;

layout(push_constant) uniform PushConstantData {
    vec4 cameraPos;
} pushConstant;

layout(set = 1,binding = 0) uniform sampler2D albedoMap;
layout(set = 1,binding = 1) uniform sampler2D metallicMap;
layout(set = 1,binding = 2) uniform sampler2D roughnessMap;   
layout(set = 1,binding = 3) uniform sampler2D normalMap;
layout(set = 1,binding = 4) uniform sampler2D aoMap;  

layout(set = 2,binding = 0) uniform ULight{
	vec4 position;
	vec4 color;
}uLight;

const float PI = 3.14159265359;

layout(location = 0) out vec4 oColor;

vec3 getNormalFromMap()
{
    vec3 tangentNormal = normalize(texture(normalMap, texCoords).xyz) * 2 - 1;

	vec3 N = normalize(normal);    
    vec3 T = normalize(tangents);   
    vec3 B = abs(cross(T, N));                 
    mat3 TBN = mat3(T, B, N);                 

    return normalize(TBN * tangentNormal);
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
	float a = roughness*roughness;
	float a2 = a*a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH*NdotH;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;
	return a2 / denom;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}



void main()
{
    const float alphaThreshold = 0.5;

    vec3  albedo    =  texture(albedoMap, texCoords).rgb;
    float metallic  = texture(metallicMap, texCoords).r;

    float roughness = texture(roughnessMap, texCoords).r;
    float ao        = texture(aoMap, texCoords).r;

    //roughness pattern
    //roughness = max(roughness, step(fract(worldPos.y * 2.02), 0.5));
    // normal
    vec3 n = getNormalFromMap();
    //vec3 n = normalize(normal);
    vec3 tangentNormal = normalize(texture(normalMap, texCoords).xyz);
    //vec3 n = normalize(tbn * tangentNormal);

    // view direction, point to the camera
    vec3 v = normalize(vec3(pushConstant.cameraPos) - worldPos);
    vec3 l = normalize(uLight.position.xyz - worldPos);
    vec3 h = normalize(v + l);  

    float ndoth = clamp(dot(n, h), 0.0 ,1.0);
    float ndotv = clamp(dot(n, v), 0.0 ,1.0);
    float ndotl = clamp(dot(n, l), 0.0 ,1.0);
    float ldoth = clamp(dot(l, h), 0.0, 1.0);
    float vdoth = dot(v, h);
    
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 LAmbient = albedo ;

    vec3 color = vec3(0.0);

    if (ndotl > 0.0){
    vec3 F = fresnelSchlick(vdoth,F0);
    //Fresnel factor

    vec3 Ldiffuse = ( albedo / PI ) * ( vec3(1) - F ) * ( 1 - metallic );
   

	float Dh =  DistributionGGX(n,h,roughness);
    //distribution

    //float ap = 2 / ( pow(roughness, 4) + 0.0001 ) - 2;
    //float Dh = ( (ap + 2) / (2 * PI) ) * max(pow( ndoth, ap),0);
   
    float Glv = min( 1 , min( 2 * ( ndoth * ndotv ) / vdoth , 2 * ( ndoth * ndotl)  / vdoth ) );
   //geometry shadowing
   
    vec3 specular = F * Dh * Glv / max(0.000001, 4.0 * ndotl * ndotv);
   
    color += (specular + Ldiffuse) * ndotl + LAmbient * 0.001;
    }
   
    if (texture(albedoMap, texCoords).a < alphaThreshold) {
        discard;
    }
    oColor = vec4(color,1.0);


}