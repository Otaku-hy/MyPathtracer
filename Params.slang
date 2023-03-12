#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

struct RenderingRuntimeParams
{
    uint frameCount = 0u;
    uint2 frameDim = uint2(0,0);
};

END_NAMESPACE_FALCOR
