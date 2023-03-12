/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "GIBufferPass.h"

const RenderPass::Info GIBufferPass::kInfo{ "GIBufferPass", "Insert pass description here." };

namespace
{
    const std::string kShaderModel = "6_5";
    const std::string kTracePassPath = "RenderPasses/GIBufferPass/TracePass.rt.slang";

    const std::string kInputVBuffer = "vbuffer";

    ChannelList InputChannel
    {
        {kInputVBuffer,"vbuffer","input vbuffer to get visible point",false},
    };

    const std::string kOutputColor = "outputColor";
    const std::string kOutputvPos = "vPosW";
    const std::string kOutputvNorm = "vNormW";
    const std::string kOutputsPos = "sPosW";
    const std::string kOutputsNorm = "sNormW";
    const std::string kOutputvColor = "vColor";
    const std::string kOutputsColor = "sColor";
    const std::string kOutputPdf = "random";

    ChannelList OutputChannel
    {
        {kOutputColor,"finalColor","the final output color",false,ResourceFormat::RGBA32Float},
        {kOutputvPos,  "gVPosW",   "Visible point",                                false,  ResourceFormat::RGBA32Float },
        {kOutputvNorm, "gVNormW",  "Visible surface normal",                       false,  ResourceFormat::RGBA32Float },
        {kOutputvColor, "gVColor",  "Outgoing radiance at visible point in RGBA",   false,  ResourceFormat::RGBA32Float },
        {kOutputsPos,  "gSPosW",   "Sample point",                                 false,  ResourceFormat::RGBA32Float },
        {kOutputsNorm, "gSNormW",  "Sample surface normal",                        false,  ResourceFormat::RGBA32Float },
        {kOutputsColor, "gSColor",  "Outgoing radiance at sample point in RGBA",    false,  ResourceFormat::RGBA32Float },
        {kOutputPdf, "gPdf",     "Random numbers used for path",                 false,  ResourceFormat::R32Float },
    };

    uint32_t kMaxPayloadSizeBytes = 128u;
}
// Don't remove this. it's required for hot-reload to function properly
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary & lib)
{
    lib.registerPass(GIBufferPass::kInfo, GIBufferPass::create);
}

GIBufferPass::GIBufferPass() : RenderPass(kInfo)
{
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);
}

void GIBufferPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mParams.frameDim = compileData.defaultTexDims;
}

GIBufferPass::SharedPtr GIBufferPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new GIBufferPass());
    return pPass;
}

Dictionary GIBufferPass::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection GIBufferPass::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, InputChannel);
    addRenderPassOutputs(reflector, OutputChannel);
    return reflector;
}

void GIBufferPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // renderData holds the requested resources
    // auto& pTexture = renderData["src"]->asTexture();
    auto& dict = renderData.getDictionary();

    BeginFrame(pRenderContext, renderData);
    if (!mpScene) return;

    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mRecompiled = true;
        mOptionsChanged = false;
    }

    PrepareProgram(pRenderContext, renderData);

    auto vars = mTracePass.mVars->getRootVar();

    vars["sampleInitializer"]["vbuffer"] = renderData[kInputVBuffer]->asTexture();
    vars["sampleInitializer"]["outputColor"] = renderData[kOutputColor]->asTexture();

    vars["pathtracer"]["params"].setBlob(mParams);

    vars["gScene"] = mpScene->getParameterBlock();

    vars["gVPosW"] = renderData[kOutputvPos]->asTexture();
    vars["gVNormW"] = renderData[kOutputvNorm]->asTexture();
    vars["gVColor"] = renderData[kOutputvColor]->asTexture();
    vars["gSPosW"] = renderData[kOutputsPos]->asTexture();
    vars["gSNormW"] = renderData[kOutputsNorm]->asTexture();
    vars["gSColor"] = renderData[kOutputsColor]->asTexture();
    vars["gPdf"] = renderData[kOutputPdf]->asTexture();

    if (mpEnvMapSampler) mpEnvMapSampler->setShaderData(vars["pathtracer"]["envMapSampler"]);
    if (mpEmissiveSampler) mpEmissiveSampler->setShaderData(vars["pathtracer"]["emissiveSampler"]);

    mpScene->raytrace(pRenderContext, mTracePass.mProgram.get(), mTracePass.mVars, uint3(mParams.frameDim.x, mParams.frameDim.y, 1u));
}

void GIBufferPass::BeginFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pOutputColor = renderData[kOutputColor]->asTexture();
    FALCOR_ASSERT(pOutputColor);

    if (!mpScene)
    {
        clearRenderPassChannels(pRenderContext, OutputChannel, renderData);
        return;
    }

    // Set output frame dimension.

    mParams.frameCount++;
    mParams.frameDim = uint2(pOutputColor->getWidth(), pOutputColor->getHeight());
}

void GIBufferPass::renderUI(Gui::Widgets& widget)
{
    bool staticDirty = false;
    bool runtimeDirty = false;

    staticDirty |= widget.var("bounces", mStParams.kMaxBounces, 0u, 10u);
    widget.tooltip("Parameter control max bounce, 0 means only direct lighting", true);

    staticDirty |= widget.var("spp", mStParams.kSamplePerPixel, 1u, 128u);
    widget.tooltip("sample per pixel");

    staticDirty |= widget.checkbox("useNEE", mStParams.kUsedNEE);
    widget.tooltip("use NEE");
    if (mStParams.kUsedNEE)
    {
        staticDirty |= widget.checkbox("useMis", mStParams.kUsedMis);
        widget.tooltip("use MIS");
    }
    staticDirty |= widget.checkbox("useReSTIRDI", mStParams.kUsedReSTIRDI);
    widget.tooltip("use RTXDI");

    staticDirty |= widget.checkbox("useThp", mStParams.kUsedThp);

    if (staticDirty) mRecompiled = true;

    bool dirty = staticDirty || runtimeDirty;

    if (dirty)
    {
        mOptionsChanged = true;
    }
}

void GIBufferPass::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
    mParams.frameCount = 0;

    mTracePass.mProgram = nullptr;
    mTracePass.mBindTable = nullptr;
    mTracePass.mVars = nullptr;

    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    if (mpScene->useEnvLight())
    {
        mpEnvMapSampler = EnvMapSampler::create(pRenderContext, mpScene->getEnvMap());
    }
    if (mpScene->useEmissiveLights())
    {
        mpEmissiveSampler = EmissiveUniformSampler::create(pRenderContext, mpScene);
    }

    RtProgram::Desc desc;
    desc.addShaderLibrary(kTracePassPath);
    desc.setShaderModel(kShaderModel);
    desc.addTypeConformances(mpScene->getTypeConformances());
    desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
    desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
    desc.setMaxTraceRecursionDepth(1);

    mTracePass.mBindTable = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
    mTracePass.mBindTable->setRayGen(desc.addRayGen("RayGen"));
    mTracePass.mBindTable->setMiss(0, desc.addMiss("ScatterMiss"));

    if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
    {
        mTracePass.mBindTable->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("ScatterTriangleClosestHit", "ScatterTriangleAnyHit"));
    }

    if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
    {
        mTracePass.mBindTable->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh), desc.addHitGroup("ScatterDisplacedTriangleMeshClosestHit", "", "DisplacedTriangleMeshIntersection"));
    }

    auto defines = GetDefines();

    mTracePass.mProgram = RtProgram::create(desc, defines);
    mTracePass.mVars = RtProgramVars::create(mTracePass.mProgram, mTracePass.mBindTable);
}

void GIBufferPass::PrepareProgram(RenderContext* pRenderComntext, const RenderData& renderData)
{
    if (!mRecompiled) return;

    auto defines = GetDefines();

    RtProgram::Desc desc = mTracePass.mProgram->getRtDesc();
    mTracePass.mProgram = RtProgram::create(desc, defines);

    mTracePass.mVars = RtProgramVars::create(mTracePass.mProgram, mTracePass.mBindTable);

    mRecompiled = false;
}

Program::DefineList GIBufferPass::GetDefines()
{
    Program::DefineList defines = {};

    if (mpSampleGenerator) defines.add(mpSampleGenerator->getDefines());
    if (mpEmissiveSampler) defines.add(mpEmissiveSampler->getDefines());

    if (mpScene)
    {
        defines.add(mpScene->getSceneDefines());
        defines.add("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
        defines.add("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
        defines.add("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    }

    defines.add("USE_RESTIRDI", mStParams.kUsedReSTIRDI ? "1" : "0");
    defines.add("USE_NEE", mStParams.kUsedNEE ? "1" : "0");
    defines.add("USE_MIS", mStParams.kUsedMis ? "1" : "0");
    defines.add("MAX_GI_BOUNCE", std::to_string(mStParams.kMaxBounces));
    defines.add("SAMPLE_PER_PIXEL", std::to_string(mStParams.kSamplePerPixel));

    defines.add("USE_Thp", mStParams.kUsedThp ? "1" : "0");

    return defines;
}

