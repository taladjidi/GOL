#include "gol_engine_vulkan.h"

#include <stdlib.h>
#include <string.h>

typedef GOLVulkanEngineParams GOLVulkanEngineImpl;

static bool vulkan_grid_valid(const GOLGrid *grid) {
    if (grid == NULL) {
        return false;
    }
    if (grid->w <= 0 || grid->h <= 0) {
        return false;
    }
    if (grid->planeCount < 2) {
        return false;
    }
    if ((grid->planeCells & 1u) != 0u) {
        return false;
    }
    return (size_t)grid->w * (size_t)grid->h <= grid->planeCells;
}

static bool vulkan_impl_valid(const GOLEngine *engine,
                              const GOLVulkanEngineImpl *impl) {
    const GOLGrid *grid;
    size_t requiredGrid;
    size_t requiredStats;

    if (engine == NULL || impl == NULL) {
        return false;
    }
    grid = engine->grid;
    if (!vulkan_grid_valid(grid)) {
        return false;
    }
    if (impl->pipeline == VK_NULL_HANDLE ||
        impl->layout == VK_NULL_HANDLE ||
        impl->descriptorSet == VK_NULL_HANDLE ||
        impl->gridBuffer == VK_NULL_HANDLE ||
        impl->statsBuffer == VK_NULL_HANDLE) {
        return false;
    }
    requiredGrid = (size_t)grid->planeCount * grid->planeCells *
                   sizeof(uint16_t);
    if (impl->gridBytes < requiredGrid) {
        return false;
    }
    requiredStats = (size_t)grid->planeCount * sizeof(GOLStats);
    if (impl->statsBytes < requiredStats) {
        return false;
    }
    return true;
}

static bool vulkan_dispatch_shape(const GOLEngine *engine,
                                  const GOLVulkanEngineImpl *impl,
                                  uint32_t *groupsX,
                                  uint32_t *groupsY) {
    const GOLGrid *grid;
    uint32_t w;
    uint32_t h;
    uint32_t tx = GOL_VULKAN_LOCAL_X;
    uint32_t ty = GOL_VULKAN_LOCAL_Y;

    if (engine == NULL || impl == NULL || groupsX == NULL || groupsY == NULL) {
        return false;
    }
    grid = engine->grid;
    if (!vulkan_grid_valid(grid)) {
        return false;
    }
    w = (uint32_t)grid->w;
    h = (uint32_t)grid->h;
    *groupsX = (w + tx - 1u) / tx;
    *groupsY = (h + ty - 1u) / ty;
    if (*groupsX == 0u || *groupsY == 0u) {
        return false;
    }
    if (impl->maxWorkGroupsX != 0u && *groupsX > impl->maxWorkGroupsX) {
        return false;
    }
    if (impl->maxWorkGroupsY != 0u && *groupsY > impl->maxWorkGroupsY) {
        return false;
    }
    return true;
}

static void vulkan_make_push_constants(const GOLEngine *engine,
                                       uint32_t readPlane,
                                       uint32_t writePlane,
                                       GOLVulkanPushConstants *pc) {
    const GOLGrid *grid;

    if (engine == NULL || pc == NULL) {
        return;
    }
    grid = engine->grid;
    pc->gridW = (uint32_t)grid->w;
    pc->gridH = (uint32_t)grid->h;
    pc->readPlane = readPlane;
    pc->writePlane = writePlane;
    pc->planeCells = (uint32_t)grid->planeCells;
    pc->birth = engine->rules.birth;
    pc->survival = engine->rules.survival;
}

static void vulkan_bind(VkCommandBuffer commandBuffer,
                        const GOLVulkanEngineImpl *impl) {
    VkDescriptorSet descriptorSet;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      impl->pipeline);
    descriptorSet = impl->descriptorSet;
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            impl->layout, 0u, 1u, &descriptorSet, 0u, NULL);
}

static bool vulkan_step_single(GOLEngine *engine, void *commandBuffer,
                               uint32_t readPlane, uint32_t writePlane) {
    GOLVulkanEngineImpl *impl;
    VkCommandBuffer cb;
    const GOLGrid *grid;
    uint32_t groupsX;
    uint32_t groupsY;
    GOLVulkanPushConstants pc;

    if (engine == NULL) {
        return false;
    }
    impl = (GOLVulkanEngineImpl *)engine->impl;
    if (!vulkan_impl_valid(engine, impl)) {
        return false;
    }
    cb = (VkCommandBuffer)commandBuffer;
    if (cb == VK_NULL_HANDLE) {
        return false;
    }
    grid = engine->grid;
    if (readPlane >= (uint32_t)grid->planeCount ||
        writePlane >= (uint32_t)grid->planeCount ||
        readPlane == writePlane) {
        return false;
    }
    if (!vulkan_dispatch_shape(engine, impl, &groupsX, &groupsY)) {
        return false;
    }

    vulkan_bind(cb, impl);
    vulkan_make_push_constants(engine, readPlane, writePlane, &pc);
    vkCmdPushConstants(cb, impl->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                       (uint32_t)sizeof(pc), &pc);
    vkCmdDispatch(cb, groupsX, groupsY, 1u);
    return true;
}

static bool vulkan_step_range(GOLEngine *engine, void *commandBuffer,
                              uint32_t startPlane, uint32_t count) {
    GOLVulkanEngineImpl *impl;
    VkCommandBuffer cb;
    const GOLGrid *grid;
    uint32_t groupsX;
    uint32_t groupsY;
    uint32_t planes;
    uint32_t i;
    size_t statsBytes;
    size_t clearBytes;

    if (engine == NULL) {
        return false;
    }
    impl = (GOLVulkanEngineImpl *)engine->impl;
    if (!vulkan_impl_valid(engine, impl)) {
        return false;
    }
    if (count == 0u) {
        return true;
    }
    cb = (VkCommandBuffer)commandBuffer;
    if (cb == VK_NULL_HANDLE) {
        return false;
    }
    grid = engine->grid;
    planes = (uint32_t)grid->planeCount;
    if (startPlane >= planes) {
        return false;
    }
    if (!vulkan_dispatch_shape(engine, impl, &groupsX, &groupsY)) {
        return false;
    }

    statsBytes = impl->statsBytes;
    clearBytes = (size_t)planes * sizeof(GOLStats);
    if (clearBytes > statsBytes) {
        clearBytes = (size_t)statsBytes;
    }
    if (impl->statsMap != NULL && clearBytes > 0u) {
        memset(impl->statsMap, 0, clearBytes);
    }

    vulkan_bind(cb, impl);
    for (i = 0u; i < count; i++) {
        uint32_t readPlane = (startPlane + i) % planes;
        uint32_t writePlane = (startPlane + i + 1u) % planes;
        GOLVulkanPushConstants pc;

        vulkan_make_push_constants(engine, readPlane, writePlane, &pc);
        vkCmdPushConstants(cb, impl->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                           (uint32_t)sizeof(pc), &pc);
        vkCmdDispatch(cb, groupsX, groupsY, 1u);

        if (i + 1u < count) {
            VkBufferMemoryBarrier barriers[2];
            VkPipelineStageFlags stage;
            VkAccessFlags access;

            memset(barriers, 0, sizeof(barriers));
            stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

            barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barriers[0].buffer = impl->gridBuffer;
            barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].offset = 0u;
            barriers[0].size = VK_WHOLE_SIZE;
            barriers[0].srcAccessMask = access;
            barriers[0].dstAccessMask = access;

            barriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barriers[1].buffer = impl->statsBuffer;
            barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[1].offset = 0u;
            barriers[1].size = VK_WHOLE_SIZE;
            barriers[1].srcAccessMask = access;
            barriers[1].dstAccessMask = access;

            vkCmdPipelineBarrier(cb, stage, stage, 0u, 0u, NULL,
                                 2u, barriers,
                                 0u, NULL);
        }
    }
    return true;
}

static void vulkan_count(GOLEngine *engine, uint32_t plane,
                         uint32_t *alive, uint32_t *maxAge) {
    GOLVulkanEngineImpl *impl;
    const GOLGrid *grid;
    uint32_t a;
    uint32_t m;

    a = 0u;
    m = 0u;
    if (engine != NULL) {
        impl = (GOLVulkanEngineImpl *)engine->impl;
        grid = engine->grid;
        if (impl != NULL && impl->statsMap != NULL && grid != NULL &&
            grid->planeCount > 0 && plane < (uint32_t)grid->planeCount) {
            const uint32_t *stats =
                (const uint32_t *)impl->statsMap +
                (size_t)plane * (sizeof(GOLStats) / sizeof(uint32_t));
            a = stats[0];
            m = stats[1];
        }
    }
    if (alive != NULL) {
        *alive = a;
    }
    if (maxAge != NULL) {
        *maxAge = m;
    }
}

static void vulkan_destroy_impl(GOLEngine *engine) {
    if (engine != NULL && engine->impl != NULL) {
        free(engine->impl);
        engine->impl = NULL;
    }
}

GOLEngine *gol_engine_create_vulkan(GOLGrid *grid,
                                    const GOLVulkanEngineParams *params) {
    GOLEngine *engine;
    GOLVulkanEngineImpl *impl;

    if (params == NULL) {
        return NULL;
    }

    engine = (GOLEngine *)malloc(sizeof(*engine));
    if (engine == NULL) {
        return NULL;
    }
    memset(engine, 0, sizeof(*engine));

    impl = (GOLVulkanEngineImpl *)malloc(sizeof(*impl));
    if (impl == NULL) {
        free(engine);
        return NULL;
    }
    *impl = *params;

    engine->grid = grid;
    engine->rules = gol_default_rules();
    engine->impl = impl;
    engine->flags = GOL_ENGINE_FLAG_GPU;
    engine->stepSingle = vulkan_step_single;
    engine->stepRange = vulkan_step_range;
    engine->count = vulkan_count;
    engine->destroyImpl = vulkan_destroy_impl;

    if (!vulkan_impl_valid(engine, impl)) {
        gol_engine_destroy(engine);
        return NULL;
    }
    return engine;
}
