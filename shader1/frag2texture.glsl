#version 330 core

in vec2 uv;
out vec4 fragColor;

uniform vec3 cameraPosition;
uniform vec2 iResolution;

struct LightFieldSurface {
    sampler2D radianceProbeGrid;
};

uniform LightFieldSurface lightFieldSurface;  // Updated uniform

void main() {
    vec2 ndc = (2.0 * gl_FragCoord.xy - iResolution) / min(iResolution.y, iResolution.x);
    vec3 rayDirView = vec3(ndc, -1.0);

    float fov = radians(60.0);
    float aspectRatio = iResolution.x / iResolution.y;

    vec3 cameraRight = vec3(1.0, 0.0, 0.0);
    vec3 cameraUp = vec3(0.0, 1.0, 0.0);

    vec3 rayDirWorld = normalize(rayDirView.x * cameraRight + rayDirView.y * cameraUp + rayDirView.z * normalize(vec3(0.0, 0.0, -1.0) - cameraPosition));
    vec3 rayOrigin = cameraPosition;

    // Example: Sample the radianceProbeGrid texture from LightFieldSurface
    vec4 radianceColor = texture(lightFieldSurface.radianceProbeGrid, vec2(0.5 * (rayDirWorld.xy + 1.0)));

    fragColor = radianceColor;


    // Combine the ray direction and radiance color
    // fragColor = vec4(0.5 * (rayDirWorld + 1.0) + radianceColor.rgb, 1.0);
}
