#version 330 core

#ifndef LightFieldProbe_glsl
#define LightFieldProbe_glsl

// #include "octahedral.glsl"

#define Point2  vec2
#define Point3  vec3
#define ProbeIndex int
#define Vector2 vec2
#define Vector3 vec3

// Enumerated value
#define TraceResult int
#define TRACE_RESULT_MISS    0
#define TRACE_RESULT_HIT     1
#define TRACE_RESULT_UNKNOWN 2

const float rayBumpEpsilon    = 0.001; // meters

struct LightFieldSurface {
    sampler2DArray radianceProbeGrid;
    sampler2DArray distanceProbeGrid;

    Point3 probePosition;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

in vec2 uv;
out vec4 fragColor;

uniform vec3 cameraPosition;
uniform vec3 lookAt;
uniform vec2 iResolution;
uniform LightFieldSurface L1;
uniform mat4 cameraToWorld;
uniform vec2 dxy;
uniform vec2 ab;

float signNotZero(float f){
  return(f >= 0.0) ? 1.0 : -1.0;
}

vec2 signNotZero(vec2 v) {
  return vec2(signNotZero(v.x), signNotZero(v.y));
}

/** 
Assumes that v is a unit vector. The result is an octahedral vector on the [-1, +1] square. 
v_cg is 
*/
vec2 octEncode(in vec3 v_cg) {
    vec3 v = vec3(v_cg.z, v_cg.x, v_cg.y);
    float l1norm = abs(v.x) + abs(v.y) + abs(v.z);
    vec2 result = v.xy * (1.0 / l1norm);
    if (v.z < 0.0) {
        result = (1.0 - abs(result.yx)) * signNotZero(result.xy);
    }
    return result;
}


vec2 oct_vec_to_uv(vec2 oct) {
    return oct.xy * 0.5 + 0.5;
}

TraceResult traceOneProbeOct(in LightFieldSurface L, in ProbeIndex index, in Ray worldSpaceRay, inout float t0, inout float t1, inout vec2 hitProbeTexCoord, inout vec3 output_log) {
    // How short of a ray segment is not worth tracing?
    const float degenreateEpsilon = 0.001; // meters
    
    Point3 probeOrigin = L.probePosition;
    
    Ray probeSpaceRay;
    probeSpaceRay.origin    = worldSpaceRay.origin - probeOrigin;
    probeSpaceRay.direction = worldSpaceRay.direction;

    if (all(equal(probeSpaceRay.origin, vec3(0.0)))) {
        // The origin of probeSpaceRay is (0, 0, 0)
        // project on to irradiance map
        // probe_space_start = ray_dirs[:, :3] * (segment_to_trace[:, 0] + rayBumpEpsilon).reshape(num_pixels, 1)
        // probe_space_end = ray_dirs[:, :3] * (segment_to_trace[:, 1] - rayBumpEpsilon).reshape(num_pixels, 1)

        Vector3 probeSpaceStartPoint = probeSpaceRay.direction * (t0 + rayBumpEpsilon);
        Vector3 probeSpaceEndPoint = probeSpaceRay.direction * (t1 - rayBumpEpsilon);
        output_log.xy = oct_vec_to_uv(octEncode(normalize(probeSpaceStartPoint)));


        hitProbeTexCoord = oct_vec_to_uv(octEncode(normalize(probeSpaceStartPoint)));
        return TRACE_RESULT_HIT;
    }

    return TRACE_RESULT_MISS;

}


bool traceProject(LightFieldSurface L, Ray worldSpaceRay, inout float tMax, out Point2 hitProbeTexCoord, out ProbeIndex hitProbeIndex, out vec3 output_log) {
    hitProbeIndex = -1;

    // TODO: replace i with nearestProbeIndices(), i with relativeProbeIndex in line 69
    int i = 0; 
    int probesLeft = 1;
    float tMin = 0.0f;

    while (probesLeft > 0) {
        TraceResult result = traceOneProbeOct(L, i, worldSpaceRay, tMin, tMax, hitProbeTexCoord, output_log);
        if (result == TRACE_RESULT_UNKNOWN) {
            // i = nextCycleIndex(i);
            --probesLeft;
        } else {
            if (result == TRACE_RESULT_HIT) {
                // TODO: hitProbeIndex = relativeProbeIndex(L, baseIndex, i);
                hitProbeIndex = i;
            }
            // Found the hit point
            break;
        }
    }

    return (hitProbeIndex != -1);
}


void main() {
    // generate ray
    float pixel_y_c = ab.y - dxy.y * (iResolution[1] - gl_FragCoord.y);
    float pixel_x_c = ab.x + dxy.x * (gl_FragCoord.x);
    vec4 lst = vec4(pixel_x_c, pixel_y_c, -1.0, 1.0);
    vec4 pixel_world = lst * cameraToWorld;
    pixel_world /= pixel_world[3];

    Ray ray;
    ray.direction = normalize(pixel_world.xyz - cameraPosition);
    ray.origin = cameraPosition;

    float   hitDistance = 15;
    Point2  hitProbeTexCoord;
    int     probeIndex;
    vec3 output_log;

//    traceProject(L1, ray, hitDistance, hitProbeTexCoord, probeIndex, output_log);

//
//    // fragColor.xyz = output_log;
//    if (!traceProject(L1, ray, hitDistance, hitProbeTexCoord, probeIndex, output_log)) {
//        // Missed the entire scene; assign black
//        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
//    } else {
//        // Sample the light probe radiance texture
//        // gl_FragColor = textureLod(L1.radianceProbeGrid, vec3(hitProbeTexCoord.x, hitProbeTexCoord.y, probeIndex), 0);
//        gl_FragColor = textureLod(L1.radianceProbeGrid, vec3(hitProbeTexCoord.x, 1.0 - hitProbeTexCoord.y, probeIndex), 0);
//        // gl_FragColor.xy = hitProbeTexCoord;
//    }

    
    
    // Example: Sample the radianceProbeGrid texture
    // vec3(uv, 0.0): layer 0.0
     vec4 radianceColor = texture(L1.radianceProbeGrid, vec3(gl_FragCoord.x / iResolution[0], 1.0 - (gl_FragCoord.y / iResolution[1]), 0.0));
     fragColor = radianceColor;

    // fragColor.xyz = ray.direction / 2 + vec3(0.5);

    // vec2 normalizedCoords = gl_FragCoord.xy / iResolution;
    // fragColor = vec4(normalizedCoords, 0.0, 1.0);

    // fragColor = pixel_world;


}


#endif // Header guard
