#include <vulkan/vulkan.h>

#include "gol.h"
#include "gol_grid.h"
#include "gol_engine_vulkan.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef VK_MAGIC_NUMBER
#define VK_MAGIC_NUMBER 0x07230203
#endif

#if VK_HEADER_VERSION >= 357
typedef VkPhysicalDeviceMemoryProperties VulkanMemoryProperties;
#else
typedef VkMemoryProperties VulkanMemoryProperties;
#endif

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    uint32_t family;
    VkPhysicalDeviceLimits limits;
    VulkanMemoryProperties memoryProps;
} VulkanTest;

typedef void (*VulkanRecordFunc)(VkCommandBuffer commandBuffer, void *user);

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

static uint32_t splitmix64(void) {
    uint64_t z;
    rng_state += 0x9E3779B97F4A7C15ULL;
    z = rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return (uint32_t)(z ^ (z >> 31));
}

static bool random_alive(double density) {
    uint32_t threshold;
    if (density <= 0.0) {
        return false;
    }
    if (density >= 1.0) {
        return true;
    }
    threshold = (uint32_t)(density * 4294967296.0);
    return splitmix64() < threshold;
}

static void destroy_context(VulkanTest *ctx) {
    if (ctx->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(ctx->device);
        vkDestroyDevice(ctx->device, NULL);
        ctx->device = VK_NULL_HANDLE;
    }
    if (ctx->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(ctx->instance, NULL);
        ctx->instance = VK_NULL_HANDLE;
    }
}

static bool init_context(VulkanTest *ctx) {
    memset(ctx, 0, sizeof(*ctx));

    VkApplicationInfo app;
    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "gol_vulkan_test";
    app.applicationVersion = 0;
    app.pEngineName = "gol";
    app.engineVersion = 0;
    app.apiVersion = VK_MAKE_VERSION(1, 0, 0);

    VkInstanceCreateInfo instanceInfo;
    memset(&instanceInfo, 0, sizeof(instanceInfo));
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &app;

    if (vkCreateInstance(&instanceInfo, NULL, &ctx->instance) != VK_SUCCESS) {
        return false;
    }

    uint32_t deviceCount = 0;
    VkResult enumerateResult = vkEnumeratePhysicalDevices(ctx->instance, &deviceCount, NULL);
    if ((enumerateResult != VK_SUCCESS && enumerateResult != VK_INCOMPLETE) ||
        deviceCount == 0) {
        vkDestroyInstance(ctx->instance, NULL);
        ctx->instance = VK_NULL_HANDLE;
        return false;
    }

    VkPhysicalDevice *devices = (VkPhysicalDevice *)malloc(deviceCount * sizeof(*devices));
    if (devices == NULL) {
        vkDestroyInstance(ctx->instance, NULL);
        ctx->instance = VK_NULL_HANDLE;
        return false;
    }

    VkResult result = vkEnumeratePhysicalDevices(ctx->instance, &deviceCount, devices);
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
        free(devices);
        vkDestroyInstance(ctx->instance, NULL);
        ctx->instance = VK_NULL_HANDLE;
        return false;
    }

    bool found = false;
    for (uint32_t i = 0; i < deviceCount && !found; i++) {
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &familyCount, NULL);
        if (familyCount == 0) {
            continue;
        }

        VkQueueFamilyProperties *families =
            (VkQueueFamilyProperties *)malloc(familyCount * sizeof(*families));
        if (families == NULL) {
            continue;
        }

        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &familyCount, families);

        uint32_t family = 0;
        bool hasCompute = false;
        for (uint32_t j = 0; j < familyCount; j++) {
            if ((families[j].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                family = j;
                hasCompute = true;
                break;
            }
        }

        if (hasCompute) {
            vkGetPhysicalDeviceMemoryProperties(devices[i], &ctx->memoryProps);

            bool hasHostCoherent = false;
            for (uint32_t m = 0; m < ctx->memoryProps.memoryTypeCount; m++) {
                uint32_t props = ctx->memoryProps.memoryTypes[m].propertyFlags;
                if ((props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 &&
                    (props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
                    hasHostCoherent = true;
                    break;
                }
            }

            if (hasHostCoherent) {
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(devices[i], &props);
                ctx->limits = props.limits;

                float priority = 1.0f;
                VkDeviceQueueCreateInfo queueInfo;
                memset(&queueInfo, 0, sizeof(queueInfo));
                queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queueInfo.queueFamilyIndex = family;
                queueInfo.queueCount = 1;
                queueInfo.pQueuePriorities = &priority;

                VkDeviceCreateInfo deviceInfo;
                memset(&deviceInfo, 0, sizeof(deviceInfo));
                deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
                deviceInfo.queueCreateInfoCount = 1;
                deviceInfo.pQueueCreateInfos = &queueInfo;

                if (vkCreateDevice(devices[i], &deviceInfo, NULL, &ctx->device) == VK_SUCCESS) {
                    ctx->physical = devices[i];
                    ctx->family = family;
                    vkGetDeviceQueue(ctx->device, family, 0, &ctx->queue);
                    found = true;
                }
            }
        }

        free(families);
    }

    free(devices);

    if (!found) {
        vkDestroyInstance(ctx->instance, NULL);
        ctx->instance = VK_NULL_HANDLE;
        return false;
    }

    return true;
}

static void destroy_pipeline(VulkanTest *ctx, VkPipeline pipeline,
                             VkPipelineLayout layout, VkDescriptorSetLayout dsl) {
    if (ctx->device == VK_NULL_HANDLE) {
        return;
    }
    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(ctx->device, pipeline, NULL);
    }
    if (layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx->device, layout, NULL);
    }
    if (dsl != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx->device, dsl, NULL);
    }
}

static bool load_spv(const char *path, uint32_t **outCode, size_t *outCount) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return false;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return false;
    }

    long length = ftell(file);
    if (length <= 0) {
        fclose(file);
        return false;
    }

    rewind(file);

    uint32_t *code = (uint32_t *)malloc((size_t)length);
    if (code == NULL) {
        fclose(file);
        return false;
    }

    size_t read = fread(code, 1, (size_t)length, file);
    fclose(file);

    if (read != (size_t)length || code[0] != VK_MAGIC_NUMBER) {
        free(code);
        return false;
    }

    *outCode = code;
    *outCount = (size_t)length / sizeof(uint32_t);
    return true;
}

static bool create_pipeline(VulkanTest *ctx, const uint32_t *code, size_t codeCount,
                            VkPipeline *pipeline, VkPipelineLayout *layout,
                            VkDescriptorSetLayout *dsl) {
    *pipeline = VK_NULL_HANDLE;
    *layout = VK_NULL_HANDLE;
    *dsl = VK_NULL_HANDLE;

    VkDescriptorSetLayoutBinding bindings[2];
    memset(bindings, 0, sizeof(bindings));

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslInfo;
    memset(&dslInfo, 0, sizeof(dslInfo));
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = 2;
    dslInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(ctx->device, &dslInfo, NULL, dsl) != VK_SUCCESS) {
        return false;
    }

    VkPushConstantRange pushRange;
    memset(&pushRange, 0, sizeof(pushRange));
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(GOLVulkanPushConstants);

    VkPipelineLayoutCreateInfo layoutInfo;
    memset(&layoutInfo, 0, sizeof(layoutInfo));
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = dsl;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(ctx->device, &layoutInfo, NULL, layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(ctx->device, *dsl, NULL);
        *dsl = VK_NULL_HANDLE;
        return false;
    }

    VkShaderModuleCreateInfo moduleInfo;
    memset(&moduleInfo, 0, sizeof(moduleInfo));
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = codeCount * sizeof(uint32_t);
    moduleInfo.pCode = code;

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx->device, &moduleInfo, NULL, &module) != VK_SUCCESS) {
        vkDestroyPipelineLayout(ctx->device, *layout, NULL);
        *layout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(ctx->device, *dsl, NULL);
        *dsl = VK_NULL_HANDLE;
        return false;
    }

    VkComputePipelineCreateInfo pipelineInfo;
    memset(&pipelineInfo, 0, sizeof(pipelineInfo));
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = module;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = *layout;

    bool ok = vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1,
                                       &pipelineInfo, NULL, pipeline) == VK_SUCCESS;
    vkDestroyShaderModule(ctx->device, module, NULL);

    if (!ok) {
        vkDestroyPipelineLayout(ctx->device, *layout, NULL);
        *layout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(ctx->device, *dsl, NULL);
        *dsl = VK_NULL_HANDLE;
    }

    return ok;
}

static VkResult map_whole_memory(VulkanTest *ctx, VkDeviceMemory memory, void **out) {
#if VK_HEADER_VERSION >= 357
    return vkMapMemory(ctx->device, memory, 0, VK_WHOLE_SIZE, 0u, out);
#else
    return vkMapMemory(ctx->device, memory, 0, VK_WHOLE_SIZE, out);
#endif
}

static bool create_buffer(VulkanTest *ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                          VkBuffer *buffer, VkDeviceMemory *memory, void **map) {
    *buffer = VK_NULL_HANDLE;
    *memory = VK_NULL_HANDLE;
    *map = NULL;

    if (size == 0) {
        return false;
    }

    VkBufferCreateInfo bufferInfo;
    memset(&bufferInfo, 0, sizeof(bufferInfo));
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(ctx->device, &bufferInfo, NULL, buffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(ctx->device, *buffer, &requirements);

    uint32_t typeIndex = 0;
    bool found = false;
    for (uint32_t i = 0; i < ctx->memoryProps.memoryTypeCount; i++) {
        if ((requirements.memoryTypeBits & (1u << i)) != 0) {
            uint32_t props = ctx->memoryProps.memoryTypes[i].propertyFlags;
            if ((props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 &&
                (props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
                typeIndex = i;
                found = true;
                break;
            }
        }
    }

    if (!found) {
        vkDestroyBuffer(ctx->device, *buffer, NULL);
        *buffer = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryAllocateInfo allocInfo;
    memset(&allocInfo, 0, sizeof(allocInfo));
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = requirements.size;
    allocInfo.memoryTypeIndex = typeIndex;

    VkDeviceMemory allocated = VK_NULL_HANDLE;
    if (vkAllocateMemory(ctx->device, &allocInfo, NULL, &allocated) != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, *buffer, NULL);
        *buffer = VK_NULL_HANDLE;
        return false;
    }

    if (vkBindBufferMemory(ctx->device, *buffer, allocated, 0) != VK_SUCCESS) {
        vkFreeMemory(ctx->device, allocated, NULL);
        vkDestroyBuffer(ctx->device, *buffer, NULL);
        *buffer = VK_NULL_HANDLE;
        return false;
    }

    void *mapped = NULL;
    if (map_whole_memory(ctx, allocated, &mapped) != VK_SUCCESS) {
        vkFreeMemory(ctx->device, allocated, NULL);
        vkDestroyBuffer(ctx->device, *buffer, NULL);
        *buffer = VK_NULL_HANDLE;
        return false;
    }

    *memory = allocated;
    *map = mapped;
    return true;
}

static void free_buffer(VulkanTest *ctx, VkBuffer buffer, VkDeviceMemory memory, void *map) {
    if (map != NULL && memory != VK_NULL_HANDLE) {
        vkUnmapMemory(ctx->device, memory);
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(ctx->device, memory, NULL);
    }
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx->device, buffer, NULL);
    }
}

static bool run_record(VulkanTest *ctx, VkCommandPool pool, VulkanRecordFunc record, void *user) {
    VkCommandBufferAllocateInfo allocInfo;
    memset(&allocInfo, 0, sizeof(allocInfo));
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(ctx->device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
        return false;
    }

    VkCommandBufferBeginInfo beginInfo;
    memset(&beginInfo, 0, sizeof(beginInfo));
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    bool ok = vkBeginCommandBuffer(commandBuffer, &beginInfo) == VK_SUCCESS;
    if (ok) {
        record(commandBuffer, user);
    }

    if (ok && vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        ok = false;
    }

    VkFenceCreateInfo fenceInfo;
    memset(&fenceInfo, 0, sizeof(fenceInfo));
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    VkFence fence = VK_NULL_HANDLE;
    if (ok && vkCreateFence(ctx->device, &fenceInfo, NULL, &fence) != VK_SUCCESS) {
        ok = false;
    }

    if (ok) {
        VkSubmitInfo submit;
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffer;

        if (vkQueueSubmit(ctx->queue, 1, &submit, fence) != VK_SUCCESS ||
            vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, 3000000000ULL) != VK_SUCCESS) {
            ok = false;
        }
    }

    if (fence != VK_NULL_HANDLE) {
        vkDestroyFence(ctx->device, fence, NULL);
    }

    vkFreeCommandBuffers(ctx->device, pool, 1, &commandBuffer);
    return ok;
}

typedef struct {
    GOLEngine *engine;
    uint32_t readPlane;
    uint32_t writePlane;
    bool ok;
} StepSingleArgs;

typedef struct {
    GOLEngine *engine;
    uint32_t startPlane;
    uint32_t count;
    bool ok;
} StepRangeArgs;

static void record_step_single(VkCommandBuffer commandBuffer, void *user) {
    StepSingleArgs *args = (StepSingleArgs *)user;
    args->ok = gol_engine_step_single(args->engine, (void *)commandBuffer,
                                      args->readPlane, args->writePlane);
}

static void record_step_range(VkCommandBuffer commandBuffer, void *user) {
    StepRangeArgs *args = (StepRangeArgs *)user;
    args->ok = gol_engine_step_range(args->engine, (void *)commandBuffer,
                                     args->startPlane, args->count);
}

static bool compare_planes(const char *label, const uint16_t *gpu, const uint16_t *cpu,
                           int w, int h) {
    size_t count = (size_t)w * (size_t)h;
    size_t mismatches = 0;

    for (size_t i = 0; i < count; i++) {
        if (gpu[i] != cpu[i]) {
            mismatches++;
            if (mismatches <= 8) {
                printf("  MISMATCH %s cell=%zu gpu=0x%04x cpu=0x%04x\n",
                       label, i, gpu[i], cpu[i]);
            }
        }
    }

    if (mismatches != 0) {
        printf("  %s: %zu mismatched cells\n", label, mismatches);
        return false;
    }

    return true;
}

static bool run_case(VulkanTest *ctx, VkPipeline pipeline, VkPipelineLayout layout,
                     VkDescriptorSetLayout dsl, VkCommandPool pool, int w, int h,
                     double density, GOLRules rules, const char *label) {
    const uint32_t planeCount = 2u;
    const size_t activeCells = (size_t)w * (size_t)h;
    const size_t planeCells = ((activeCells + 1) / 2) * 2;

    const VkDeviceSize gridBytes =
        (VkDeviceSize)planeCount * planeCells * sizeof(uint16_t);
    const VkDeviceSize statsBytes = (VkDeviceSize)planeCount * sizeof(GOLStats);

    bool ok = true;

    VkBuffer gridBuffer = VK_NULL_HANDLE;
    VkDeviceMemory gridMemory = VK_NULL_HANDLE;
    void *gridMap = NULL;

    VkBuffer statsBuffer = VK_NULL_HANDLE;
    VkDeviceMemory statsMemory = VK_NULL_HANDLE;
    void *statsMap = NULL;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    GOLGrid *grid = NULL;
    GOLEngine *engine = NULL;

    uint16_t *cpuCur = (uint16_t *)calloc(activeCells, sizeof(uint16_t));
    uint16_t *cpuNext = (uint16_t *)calloc(activeCells, sizeof(uint16_t));

    if (cpuCur == NULL || cpuNext == NULL) {
        free(cpuCur);
        free(cpuNext);
        return false;
    }

    ok = create_buffer(ctx, gridBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       &gridBuffer, &gridMemory, &gridMap);
    if (ok) {
        ok = create_buffer(ctx, statsBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           &statsBuffer, &statsMemory, &statsMap);
    }

    if (ok) {
        VkDescriptorPoolSize sizes[2];
        memset(sizes, 0, sizeof(sizes));
        sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[0].descriptorCount = 1;
        sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[1].descriptorCount = 1;

        VkDescriptorPoolCreateInfo poolInfo;
        memset(&poolInfo, 0, sizeof(poolInfo));
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = sizes;

        ok = vkCreateDescriptorPool(ctx->device, &poolInfo, NULL, &descriptorPool) == VK_SUCCESS;

        if (ok) {
            VkDescriptorSetAllocateInfo setInfo;
            memset(&setInfo, 0, sizeof(setInfo));
            setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            setInfo.descriptorPool = descriptorPool;
            setInfo.descriptorSetCount = 1;
            setInfo.pSetLayouts = &dsl;

            ok = vkAllocateDescriptorSets(ctx->device, &setInfo, &descriptorSet) == VK_SUCCESS;
        }

        if (ok) {
            VkDescriptorBufferInfo bufferInfos[2];
            memset(bufferInfos, 0, sizeof(bufferInfos));

            bufferInfos[0].buffer = gridBuffer;
            bufferInfos[0].offset = 0;
            bufferInfos[0].range = VK_WHOLE_SIZE;

            bufferInfos[1].buffer = statsBuffer;
            bufferInfos[1].offset = 0;
            bufferInfos[1].range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet writes[2];
            memset(writes, 0, sizeof(writes));

            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = descriptorSet;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].pBufferInfo = &bufferInfos[0];

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = descriptorSet;
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[1].pBufferInfo = &bufferInfos[1];

            vkUpdateDescriptorSets(ctx->device, 2, writes, 0, NULL);
        }
    }

    if (ok) {
        memset(gridMap, 0, gridBytes);
        memset(statsMap, 0, statsBytes);

        grid = gol_grid_alloc();
        if (grid == NULL) {
            ok = false;
        } else {
            gol_grid_wrap(grid, w, h, planeCells, (int)planeCount,
                          (uint16_t *)gridMap, NULL);

            GOLVulkanEngineParams params;
            memset(&params, 0, sizeof(params));
            params.pipeline = pipeline;
            params.layout = layout;
            params.descriptorSet = descriptorSet;
            params.gridBuffer = gridBuffer;
            params.gridBytes = gridBytes;
            params.gridMap = gridMap;
            params.statsBuffer = statsBuffer;
            params.statsBytes = statsBytes;
            params.statsMap = statsMap;
            params.maxWorkGroupsX = ctx->limits.maxComputeWorkGroupCount[0];
            params.maxWorkGroupsY = ctx->limits.maxComputeWorkGroupCount[1];

            engine = gol_engine_create_vulkan(grid, &params);
            if (engine == NULL) {
                ok = false;
            } else {
                gol_engine_set_rules(engine, rules);
            }
        }
    }

    if (ok) {
        uint16_t *gpu0 = (uint16_t *)gridMap;
        for (size_t i = 0; i < activeCells; i++) {
            uint16_t value = random_alive(density) ? 1u : 0u;
            gpu0[i] = value;
            cpuCur[i] = value;
        }
    }

    uint32_t readPlane = 0u;
    uint32_t writePlane = 1u;

    for (int step = 0; step < 6 && ok; step++) {
        memset(statsMap, 0, statsBytes);

        StepSingleArgs args;
        args.engine = engine;
        args.readPlane = readPlane;
        args.writePlane = writePlane;
        args.ok = false;

        if (!run_record(ctx, pool, record_step_single, &args) || !args.ok) {
            ok = false;
            break;
        }

        const uint16_t *gpuWrite =
            (const uint16_t *)gridMap + (size_t)writePlane * planeCells;

        if (!compare_planes(label, gpuWrite, cpuNext, w, h)) {
            ok = false;
            break;
        }

        int cpuAlive = 0;
        int cpuMaxAge = 0;
        gol_count_alive(cpuNext, w, h, &cpuAlive, &cpuMaxAge);

        const GOLStats *stats = (const GOLStats *)statsMap + writePlane;
        if (stats->alive != (uint32_t)cpuAlive || stats->maxAge != (uint32_t)cpuMaxAge) {
            printf("  %s step=%d stats gpu alive=%u maxAge=%u cpu alive=%d maxAge=%d\n",
                   label, step, stats->alive, stats->maxAge, cpuAlive, cpuMaxAge);
            ok = false;
            break;
        }

        uint32_t engineAlive = 0;
        uint32_t engineMaxAge = 0;
        gol_engine_count(engine, writePlane, &engineAlive, &engineMaxAge);
        if (engineAlive != stats->alive || engineMaxAge != stats->maxAge) {
            printf("  %s step=%d engine count differs from mapped stats\n", label, step);
            ok = false;
            break;
        }

        gol_step_cpu(cpuCur, cpuNext, w, h, rules);

        uint16_t *tmp = cpuCur;
        cpuCur = cpuNext;
        cpuNext = tmp;

        writePlane = readPlane;
        readPlane = (readPlane + 1u) % planeCount;
    }

    if (ok) {
        const uint32_t startPlane = readPlane;

        StepRangeArgs args;
        args.engine = engine;
        args.startPlane = startPlane;
        args.count = 3u;
        args.ok = false;

        if (!run_record(ctx, pool, record_step_range, &args) || !args.ok) {
            ok = false;
        }

        for (int i = 0; i < 3 && ok; i++) {
            gol_step_cpu(cpuCur, cpuNext, w, h, rules);
            uint16_t *tmp = cpuCur;
            cpuCur = cpuNext;
            cpuNext = tmp;
        }

        if (ok) {
            const uint32_t currentPlane = (startPlane + 3u) % planeCount;
            const uint16_t *gpuCurrent =
                (const uint16_t *)gridMap + (size_t)currentPlane * planeCells;

            if (!compare_planes(label, gpuCurrent, cpuCur, w, h)) {
                ok = false;
            }
        }
    }

    if (engine != NULL) {
        gol_engine_destroy(engine);
    }
    if (grid != NULL) {
        gol_grid_free(grid);
    }

    if (descriptorSet != VK_NULL_HANDLE && descriptorPool != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(ctx->device, descriptorPool, 1, &descriptorSet);
    }
    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx->device, descriptorPool, NULL);
    }

    free_buffer(ctx, gridBuffer, gridMemory, gridMap);
    free_buffer(ctx, statsBuffer, statsMemory, statsMap);

    free(cpuCur);
    free(cpuNext);

    return ok;
}

int main(int argc, char *argv[]) {
    printf("=== Vulkan Implementation Tests ===\n\n");

    const char *spvPath = (argc > 1) ? argv[1] : "build/gol_step.spv";

    VulkanTest ctx;
    if (!init_context(&ctx)) {
        printf("Vulkan compute device unavailable; skipping Vulkan tests\n");
        return 0;
    }

    uint32_t *spvCode = NULL;
    size_t spvCount = 0;
    if (!load_spv(spvPath, &spvCode, &spvCount)) {
        fprintf(stderr, "failed to load SPIR-V module: %s\n", spvPath);
        destroy_context(&ctx);
        return 1;
    }

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;

    if (!create_pipeline(&ctx, spvCode, spvCount, &pipeline, &layout, &dsl)) {
        fprintf(stderr, "failed to create Vulkan compute pipeline\n");
        free(spvCode);
        destroy_context(&ctx);
        return 1;
    }

    VkCommandPoolCreateInfo poolInfo;
    memset(&poolInfo, 0, sizeof(poolInfo));
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = ctx.family;

    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(ctx.device, &poolInfo, NULL, &pool) != VK_SUCCESS) {
        fprintf(stderr, "failed to create command pool\n");
        destroy_pipeline(&ctx, pipeline, layout, dsl);
        free(spvCode);
        destroy_context(&ctx);
        return 1;
    }

    const GOLRules rules[] = {
        { (1u << 3), (1u << 2) | (1u << 3) },
        { (1u << 3) | (1u << 6), (1u << 2) | (1u << 3) },
        { (1u << 3) | (1u << 6) | (1u << 8), (1u << 3) | (1u << 4) | (1u << 5) },
        { (1u << 1), (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) },
        { (1u << 3) | (1u << 8), (1u << 3) | (1u << 4) | (1u << 8) },
    };

    const int widths[] = { 16, 32, 48, 7 };
    const int heights[] = { 16, 32, 31, 5 };
    const double densities[] = { 0.1, 0.4, 0.8 };

    const size_t ruleCount = sizeof(rules) / sizeof(rules[0]);
    const size_t widthCount = sizeof(widths) / sizeof(widths[0]);
    const size_t heightCount = sizeof(heights) / sizeof(heights[0]);
    const size_t densityCount = sizeof(densities) / sizeof(densities[0]);

    bool failed = false;

    for (size_t ri = 0; ri < ruleCount && !failed; ri++) {
        for (size_t wi = 0; wi < widthCount && !failed; wi++) {
            for (size_t hi = 0; hi < heightCount && !failed; hi++) {
                for (size_t di = 0; di < densityCount && !failed; di++) {
                    char label[96];
                    snprintf(label, sizeof(label),
                             "rule=%zu grid=%dx%d density=%.2f",
                             ri, widths[wi], heights[hi], densities[di]);

                    printf("Test: %s\n", label);

                    if (!run_case(&ctx, pipeline, layout, dsl, pool,
                                  widths[wi], heights[hi], densities[di],
                                  rules[ri], label)) {
                        failed = true;
                    } else {
                        printf("  PASS\n");
                    }
                }
            }
        }
    }

    vkDestroyCommandPool(ctx.device, pool, NULL);
    destroy_pipeline(&ctx, pipeline, layout, dsl);
    free(spvCode);
    destroy_context(&ctx);

    if (failed) {
        fprintf(stderr, "\nVulkan tests failed\n");
        return 1;
    }

    printf("\nAll Vulkan tests passed!\n");
    return 0;
}
