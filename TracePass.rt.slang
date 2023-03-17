#include "Scene/ScenePrimitiveDefines.slangh"
#include "Scene/Material/MaterialDefines.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Scene.RaytracingInline;
import Scene.Shading;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Materials.MaterialShading;
import Params;
import StaticParameters;
import LoadShadingData;
import PathTracer;

RWTexture2D<float4> gVPosW;
RWTexture2D<float4> gVNormW;
RWTexture2D<float4> gVColor;
RWTexture2D<float4> gSPosW;
RWTexture2D<float4> gSNormW;
RWTexture2D<float4> gSColor;
RWTexture2D<float>  gPdf;

struct SampleInitializer
{
    Texture2D<PackedHitInfo> vbuffer;
    RWTexture2D<float3> outputColor;

    void TraceScatterRay(inout PathPayLoad pathState)
    {
        Ray ray = pathState.GenerateScatterRay();
        TraceRay(gScene.rtAccel, RAY_FLAG_NONE, 0xff, 0, rayTypeCount, 0, ray.toRayDesc(), pathState);
    };

    void TracePass(uint2 pixel, out PathSample sample)
    {
        sample = {};
        HitInfo hit = HitInfo(vbuffer[pixel]);

        float3 primaryRayDir = gScene.camera.computeRayPinhole(pixel,pathtracer.params.frameDim).dir;

        if(hit.isValid())
        {
            const float lod = 0.f;
            ShadingData sd = LoadShadingData(hit,primaryRayDir,lod,true);

            float weight = 1.f / kSamplePerPixel;

            for(int i=0;i<kSamplePerPixel;i++)
            {
                SampleGenerator sg = SampleGenerator(pixel,pathtracer.params.frameCount * kSamplePerPixel + i);

                PathPayLoad pathState = PathPayLoad(sg);
                pathState.direction = primaryRayDir;
                pathState.pdf = 1;

                pathtracer.HandleHit(pathState,hit);

                float3 thp = pathState.thp;

                sample.vPos = sd.posW;
                sample.vNorm = pathState.norm;
                sample.random = pathState.pdf;

                pathState.thp = 1.f;
                pathState.pdf = 1.f;
                pathState.radiance = 0.f;

                while(!pathState.isTerminated)
                {
                    TraceScatterRay(pathState);
                    if(pathState.currentVertexIndex == 2u)
                    {
                        sample.sPos = pathState.origin;
                        sample.sNorm = pathState.norm;
                    }
                }

                float3 wo = normalize(sample.sPos - sample.vPos);
                sample.sColor += pathState.radiance * weight;
                sample.vColor += pathState.lForDelta * thp;
                if(kUsedThp)
                {
                    sample.vColor += sample.sColor * thp;
                }
                else
                {
                    float invPdf = evalPdfBSDF(sd, wo) > 0.f ? 1.f / evalPdfBSDF(sd, wo) : 0.f;
                    sample.vColor += evalBSDFCosine(sd, wo) * sample.sColor * invPdf;
                }
            }

            outputColor[pixel] = sample.vColor;
        }
        else
        {
            float3 color = 0.f;
            //if(!kUsedReSTIRDI)
                color = pathtracer.GetBackGroundColor(primaryRayDir);
            outputColor[pixel] = color;
            sample.vColor = color;
        }

    }

};

ParameterBlock<SampleInitializer> sampleInitializer;

[shader("miss")]
void ScatterMiss(inout PathPayLoad pathState)
{
    pathtracer.HandleMiss(pathState);
}

[shader("anyhit")]
void ScatterTriangleAnyHit(inout PathPayLoad pathState, BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void ScatterTriangleClosestHit(inout PathPayLoad pathState,BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit thit;
    thit.instanceID = getGeometryInstanceID();
    thit.primitiveIndex = PrimitiveIndex();
    thit.barycentrics = attribs.barycentrics;
    HitInfo hit = HitInfo(thit);

    pathState.currentVertexIndex++;
    pathtracer.HandleHit(pathState,hit);   
}

[shader("intersection")]
void DisplacedTriangleMeshIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    DisplacedTriangleMeshIntersector::Attribs attribs;
    float t;
    if (DisplacedTriangleMeshIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void ScatterDisplacedTriangleMeshClosestHit(inout PathPayLoad pathState, DisplacedTriangleMeshIntersector::Attribs attribs)
{
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;

    HitInfo hit = HitInfo(displacedTriangleHit);
    pathState.currentVertexIndex++;
    pathtracer.HandleHit(pathState,hit);  
}

[shader("raygeneration")]
void RayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    PathSample sample = {};
    sampleInitializer.TracePass(pixel,sample);

    gVPosW[pixel] = float4(sample.vPos,1.f);
    gVNormW[pixel] = float4(sample.vNorm,0.f);
    gVColor[pixel] = float4(sample.vColor,1.f);
    gSPosW[pixel] = float4(sample.sPos,1.f);
    gSNormW[pixel] = float4(sample.sNorm,0.f);
    gSColor[pixel] = float4(sample.sColor,1.f);
    gPdf[pixel] = sample.random;
}
