#ifndef GOL_ENGINE_VULKAN_H
#define GOL_ENGINE_VULKAN_H

#include "gol_engine.h"

#include <vulkan/vulkan.h>

#define GOL_VULKAN_LOCAL_X 8u
#define GOL_VULKAN_LOCAL_Y 8u

typedef struct {
    uint32_t gridW;
    uint32_t gridH;
    uint32_t readPlane;
    uint32_t writePlane;
    uint32_t planeCells;
    uint32_t birth;
    uint32_t survival;
} GOLVulkanPushConstants;

typedef struct {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSet descriptorSet;
    VkBuffer gridBuffer;
    VkDeviceSize gridBytes;
    void *gridMap;
    VkBuffer statsBuffer;
    VkDeviceSize statsBytes;
    void *statsMap;
    uint32_t maxWorkGroupsX;
    uint32_t maxWorkGroupsY;
} GOLVulkanEngineParams;

GOLEngine *gol_engine_create_vulkan(GOLGrid *grid,
                                    const GOLVulkanEngineParams *params);

#endif
