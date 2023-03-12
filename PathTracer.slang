#include "Scene/SceneDefines.slangh"

__exported import Scene.shading;
__exported import Scene.Raytracing;
__exported import Scene.Intersection;
__exported import Scene.RaytracingInline;
__exported import Scene.Material.ShadingUtils;
__exported import Utils.Math.MathHelpers;
__exported import Utils.Geometry.GeometryHelpers;
__exported import Utils.Sampling.SampleGenerator;
__exported import Rendering.Lights.LightHelpers;
__exported import Rendering.Lights.EnvMapSampler;
__exported import Rendering.Lights.EmissiveLightSampler;
__exported import Rendering.Lights.EmissiveLightSamplerHelpers;
__exported import Params;
__exported import StaticParameters;

static const float kRayTMax = 1e20f;

ShadingData LoadShadingData(HitInfo hit,float3 rayDir,ITextureSampler lod,bool isPrimaryHit)
{
    VertexData vd = {};
    uint materialID;
        
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        vd = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }

    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        vd = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }   

    ShadingData sd = gScene.materials.prepareShadingData(vd,materialID,-rayDir,lod);

    if(isPrimaryHit)
        adjustShadingNormal(sd, vd);

    return sd;
}

struct PathSample
{
    float3 vPos;
    float3 vNorm;
    float3 vColor;
    float3 sPos;
    float3 sNorm;
    float3 sColor;
    float random;
};

struct PathPayLoad
{
    uint currentVertexIndex;             //vertexIndex along the path, intialized value is 1

    bool isTerminated;
    bool isLastVertexSampledLight;
    bool isLightSamplable;

    float3 origin;
    float3 norm;
    float3 direction;
    float3 posW;

    float pdf;
    float3 radiance;
    float3 thp;
    
    SampleGenerator sg;

    __init(SampleGenerator sampleGenerator)
    {
       sg = sampleGenerator;

       isTerminated = false;

       pdf = 0.f;
       radiance = 0.f;
       thp = 1.f;

       origin = 0.f;
       norm = 0.f;
       direction = 0.f;

       currentVertexIndex = 1u;
       isLastVertexSampledLight = false;
    }

    Ray GenerateScatterRay()
    {
        return Ray(origin,direction,0.f,kRayTMax);
    }
};

struct PathTracer
{
    static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
    static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
    static const bool kUseEnvLight = USE_ENV_LIGHT;

    EnvMapSampler envMapSampler;            
    EmissiveLightSampler emissiveSampler;

    RenderingRuntimeParams params;

    enum class LightType
    {
        EnvMap,
        Emissive,
        Analytic
    };

    struct LightSample
    {
        float3 Li;
        float pdf;
        float3 origin;
        float distance;
        float3 dir;

        uint lightType;

        Ray GetVisibilityRay()
        {
            return Ray(origin,dir,0,distance);
        }
    };

    struct PathVertex
    {
        float3 pos;
        float3 normal;
        float3 faceNormal;

        __init(float3 pos,float3 normal,float3 faceNormal)
        {
            this.pos = pos;
            this.normal = normal;
            this.faceNormal = faceNormal;
        }

        float3 GetRayOrigin(float3 rayDir)
        {
            return computeRayOrigin(pos,dot(faceNormal,rayDir) >=0 ? faceNormal : -faceNormal);
        }
    };

    float3 GetBackGroundColor(float3 dir)
    {
        float3 color = {};
        if(kUseEnvLight) color = gScene.envMap.eval(dir);
        return color;
    }

    bool HasFinishedScatter(PathPayLoad pathState)
    {
        return pathState.currentVertexIndex-1 >= kMaxBounces;
    }

    void AddToPathContribution(inout PathPayLoad pathState,float3 radiance)
    {
        pathState.radiance += pathState.thp * radiance;
    }

    float EvalMisWeight(int count0,float pdf0,int count1,float pdf1)
    {
        float w0 = count0 * pdf0;
        float w1 = count1 * pdf1;

        return w0 / (w0 + w1);
    }

    void GetLightProbabilities(out float p[3])
    {
        p[0] = kUseEnvLight ? 1.f: 0;
        p[1] = kUseEmissiveLights ? 1.f : 0;
        p[2] = kUseAnalyticLights ? 1.f : 0;
        
        float sum = p[0] + p[1] + p[2];
        if (sum == 0.f) return;

        float invsum = 1.f / sum;
        p[0] *= invsum;
        p[1] *= invsum;
        p[2] *= invsum;
    }

    float GetEnvLightProbabilities() {float p[3]; GetLightProbabilities(p); return p[0];}
    float GetEmissiveLightProbabilities() {float p[3]; GetLightProbabilities(p); return p[1];}
    float GetAnalyticLightProbabilities() {float p[3]; GetLightProbabilities(p); return p[2];}
    
    bool SelectLightType(out uint lightType, out float pdf,inout SampleGenerator sg)
    {
        float p[3];
        GetLightProbabilities(p);

        float u = sampleNext1D(sg);

        for(lightType=0;lightType<3;lightType++)
        {
            if(u < p[lightType])
            {
                pdf = p[lightType];
                return true;
            }
            u -= p[lightType];
        }
        return false;
    }

    bool GenerateEnvLightSample(PathVertex vertex,inout SampleGenerator sg,out LightSample ls)
    {
        ls = {};
        if(!kUseEnvLight) return false;
        EnvMapSample lightSample;
        if(!envMapSampler.sample(sampleNext2D(sg),lightSample)) return false;
        
        ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
        ls.pdf = lightSample.pdf;
        ls.origin = vertex.GetRayOrigin(lightSample.dir);
        ls.distance = kRayTMax;
        ls.dir = lightSample.dir;

        return any(ls.Li > 0.f);
    }

    bool GenerateEmissiveLightSample(PathVertex vertex,inout SampleGenerator sg,out LightSample ls)
    {
        ls = {};
        if(!kUseEmissiveLights) return false;
        TriangleLightSample lightSample;
        if(!emissiveSampler.sampleLight(vertex.pos,vertex.normal,true,sg,lightSample)) return false;

        ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
        ls.pdf = lightSample.pdf;

        float3 lightPos = computeRayOrigin(lightSample.posW,lightSample.normalW);
        ls.origin = vertex.GetRayOrigin(lightPos - vertex.pos);
        float3 toLight = lightPos - ls.origin;
        ls.distance = length(toLight);
        ls.dir = normalize(toLight);

        return true;
    }

    bool GenerateAnalyticLightSample(PathVertex vertex,inout SampleGenerator sg,out LightSample ls)
    {
        ls = {};

        uint lightCount = gScene.getLightCount();
        if(!kUseAnalyticLights || lightCount == 0) return false;

        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);
        AnalyticLightSample lightSample;
        if (!sampleLight(vertex.pos, gScene.getLight(lightIndex), sg, lightSample)) return false;

        ls.pdf = lightSample.pdf / lightCount;
        ls.Li = lightSample.Li * lightCount;
        ls.origin = vertex.GetRayOrigin(lightSample.dir);
        ls.distance = lightSample.distance;
        ls.dir = lightSample.dir;

        return any(ls.Li > 0.f);
    }

    bool GenerateLightSample(PathVertex vertex, SampleGenerator sg,out LightSample ls)
    {
        ls = {};
        uint lightType;
        float selectedPdf;
        bool valid = false;

        if(!SelectLightType(lightType, selectedPdf, sg)) return false;

        if(kUseEnvLight && lightType == uint(LightType::EnvMap))
            valid = GenerateEnvLightSample(vertex,sg,ls);
        if(kUseEmissiveLights && lightType == uint(LightType::Emissive))
            valid = GenerateEmissiveLightSample(vertex,sg,ls);
        if(kUseAnalyticLights && lightType == uint(LightType::Analytic))
            valid = GenerateAnalyticLightSample(vertex,sg,ls);

        if(!valid) return false;

        ls.lightType = lightType;
        ls.pdf *= selectedPdf;
        ls.Li /= selectedPdf;

        return true;
    }

    bool GenerateScatterRay(ShadingData sd,IBSDF bsdf,ITextureSampler lod,float3 rayOrigin,inout PathPayLoad pathState)
    {
        BSDFSample bsdfSample;
        if(bsdf.sample(sd,pathState.sg,bsdfSample))
        {
            pathState.thp *= bsdfSample.weight;
            pathState.direction = bsdfSample.wo;
            pathState.origin = rayOrigin;
            pathState.pdf = bsdfSample.pdf;
            return any(pathState.thp > 0.f);
        }
        return false;
    }

   
    void HandleHit(inout PathPayLoad pathState,HitInfo hit)
    {
        const HitType hitType = hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;
        const bool isPrimaryHit = (pathState.currentVertexIndex == 1);

        let lod = ExplicitLodTextureSampler(0.f);
        ShadingData sd = LoadShadingData(hit,pathState.direction,lod,isPrimaryHit);
        let bsdf = gScene.materials.getBSDF(sd, lod);

        pathState.norm = sd.N;
        pathState.posW = sd.posW;

        BSDFProperties bsdfProperties = bsdf.getProperties(sd);

        bool computeEmissive = isPrimaryHit || (!kUsedNEE || kUsedMis || !pathState.isLastVertexSampledLight);
        if(kUsedReSTIRDI && pathState.currentVertexIndex == 2) computeEmissive = false;

        if(computeEmissive && any(bsdfProperties.emission > 0.f))
        {
            float misWeight = 1.f;
            if(!isPrimaryHit && isTriangleHit && kUsedMis && pathState.isLastVertexSampledLight)
            {
                TriangleHit thit = hit.getTriangleHit();
                TriangleLightHit tlhit;
                tlhit.triangleIndex = gScene.lightCollection.getTriangleIndex(thit.instanceID, thit.primitiveIndex);
                tlhit.posW = sd.posW;
                tlhit.normalW = sd.frontFacing ? sd.faceN : -sd.faceN;

                float lightPdf = GetEmissiveLightProbabilities() * emissiveSampler.evalPdf(pathState.origin, pathState.norm, true, tlhit);
                misWeight = EvalMisWeight(1,pathState.pdf,1,lightPdf);    
            }
            AddToPathContribution(pathState,bsdfProperties.emission * misWeight);
        }

        /// terminate path with mis
        if(pathState.currentVertexIndex > kMaxBounces+1)
        {
            pathState.isTerminated = true;
            return;
        }

        uint lobes = bsdf.getLobes(sd);
        bool hasNonDeltaLobes = (lobes & (uint)LobeType::NonDelta) != 0;

        bool applyNEE = kUsedNEE && hasNonDeltaLobes;

        if(kUsedReSTIRDI && pathState.currentVertexIndex == 1) applyNEE = false;

        if(applyNEE)
        {
            PathVertex vertex = PathVertex(sd.posW,sd.N,sd.faceN);
            bool isValidSample = false;
            LightSample ls = {};

            pathState.isLastVertexSampledLight = (lobes & (uint)LobeType::NonDeltaReflection) != 0;
            if(pathState.isLastVertexSampledLight)
                isValidSample = GenerateLightSample(vertex,pathState.sg,ls);

            if(isValidSample)
            {
                if(kUsedMis && ls.lightType != uint(LightType::Analytic))
                {
                    float scatterPdf = bsdf.evalPdf(sd,ls.dir);
                    ls.Li *= EvalMisWeight(1,ls.pdf,1,scatterPdf);
                }
                float3 weight = bsdf.eval(sd,ls.dir,pathState.sg);
                float3 Lr = ls.Li * weight;
                if(any(Lr > 0.f))
                {
                    Ray ray = ls.GetVisibilityRay();
                    SceneRayQuery<1> scene_ray_query;
                    bool visible = scene_ray_query.traceVisibilityRay(ray, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff);
                    if(visible) AddToPathContribution(pathState,Lr);
                }
            }
        }

        float3 rayOrigin = sd.computeNewRayOrigin();

        bool valid = GenerateScatterRay(sd,bsdf,lod,rayOrigin,pathState);

        if(HasFinishedScatter(pathState) && kUsedNEE && !kUsedMis && pathState.isLastVertexSampledLight) valid = false;

        if(!valid) pathState.isTerminated = true;
    }

    void HandleMiss(inout PathPayLoad pathState)
    {
        pathState.isTerminated = true;
        
        bool computeEnvironment = !kUsedNEE || kUsedMis || !pathState.isLastVertexSampledLight;
        if(kUsedReSTIRDI && pathState.currentVertexIndex == 1) computeEnvironment = false;

        if(kUseEnvLight && computeEnvironment)
        {
            float3 envRadiance  = envMapSampler.eval(pathState.direction);
            float misWeight = 1.f;

            if(kUsedNEE && kUsedMis && pathState.isLastVertexSampledLight)
            {
                float lightPdf = GetEnvLightProbabilities() * envMapSampler.evalPdf(pathState.direction); 
                misWeight = EvalMisWeight(1,pathState.pdf,1,lightPdf);
            }

            AddToPathContribution(pathState,envRadiance*misWeight);
        }
    }
};

ParameterBlock<PathTracer> pathtracer;
