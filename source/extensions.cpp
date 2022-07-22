#include "../include/backend.hpp"

// Super loading Vulkan extensions
static PFN_vkGetBufferDeviceAddressKHR			__vkGetBufferDeviceAddressKHR = nullptr;
static PFN_vkGetAccelerationStructureDeviceAddressKHR	__vkGetAccelerationStructureDeviceAddressKHR = nullptr;
static PFN_vkGetAccelerationStructureBuildSizesKHR	__vkGetAccelerationStructureBuildSizesKHR = nullptr;
static PFN_vkCmdBuildAccelerationStructuresKHR		__vkCmdBuildAccelerationStructuresKHR = nullptr;

// Vulkan extensions
VKAPI_ATTR VkDeviceAddress VKAPI_CALL vkGetBufferDeviceAddressKHR
		(VkDevice device,
		const VkBufferDeviceAddressInfoKHR *pInfo)
{
	return __vkGetBufferDeviceAddressKHR(device, pInfo);
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL vkGetAccelerationStructureDeviceAddressKHR
		(VkDevice device,
		const VkAccelerationStructureDeviceAddressInfoKHR *pInfo)
{
	return __vkGetAccelerationStructureDeviceAddressKHR(device, pInfo);
}

VKAPI_ATTR void VKAPI_CALL vkGetAccelerationStructureBuildSizesKHR
		(VkDevice device,
		VkAccelerationStructureBuildTypeKHR type,
		const VkAccelerationStructureBuildGeometryInfoKHR *pInfo,
		const uint32_t *pMaxPrimCount,
		VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo)
{
	__vkGetAccelerationStructureBuildSizesKHR(device, type, pInfo, pMaxPrimCount, pSizeInfo);
}

VKAPI_ATTR void VKAPI_CALL vkCmdBuildAccelerationStructuresKHR
		(VkCommandBuffer commandBuffer,
		uint32_t infoCount,
		const VkAccelerationStructureBuildGeometryInfoKHR *pInfo,
		const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
	__vkCmdBuildAccelerationStructuresKHR(commandBuffer, infoCount, pInfo, ppBuildRangeInfos);
}

namespace kobra {

// Load Vulkan extensions
void load_vulkan_extensions(const vk::raii::Device &device)
{
	__vkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)
		vkGetDeviceProcAddr(*device, "vkGetBufferDeviceAddressKHR");
	__vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)
		vkGetDeviceProcAddr(*device, "vkGetAccelerationStructureDeviceAddressKHR");
	__vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)
		vkGetDeviceProcAddr(*device, "vkGetAccelerationStructureBuildSizesKHR");
	__vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)
		vkGetDeviceProcAddr(*device, "vkCmdBuildAccelerationStructuresKHR");
}

}
