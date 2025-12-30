use std::ffi::CStr;
use std::sync::Arc;

use ash::{vk, khr};

#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceExtensions {
    pub khr_swapchain: bool,
    pub khr_dynamic_rendering: bool,
    pub khr_portability_subset: bool,

    pub ray_tracing: bool,
}

impl DeviceExtensions {
    pub fn to_vec(&self) -> Vec<&CStr> {
        let mut result = Vec::new();

        if self.khr_swapchain {
            result.push(khr::swapchain::NAME);
        }

        if self.khr_dynamic_rendering {
            result.push(khr::dynamic_rendering::NAME)
        }

        if self.khr_portability_subset {
            result.push(khr::portability_subset::NAME)
        }

        if self.ray_tracing {
            result.push(khr::ray_query::NAME);
            result.push(khr::acceleration_structure::NAME);
            result.push(khr::deferred_host_operations::NAME);
        }

        return result
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceFeatures {
    pub sampler_anisotropy: bool,
    pub dynamic_rendering_enabled: bool,
    pub buffer_device_address: bool,

    pub ray_tracing: bool,
    pub acceleration_structure_host_commands: bool,
}

impl DeviceFeatures {
    pub fn get_core_features(&self) -> vk::PhysicalDeviceFeatures {
        vk::PhysicalDeviceFeatures::default()
            .sampler_anisotropy(self.sampler_anisotropy)
    }
}

#[derive(Default, Clone)]
pub struct ExtensionLoaders {
    pub khr_swapchain: Option<Arc<khr::swapchain::Device>>,
    pub khr_acceleration_structure: Option<Arc<khr::acceleration_structure::Device>>,
    pub khr_deferred_host_operations: Option<Arc<khr::deferred_host_operations::Device>>,
}

impl ExtensionLoaders {
    pub fn from_extensions(instance: &ash::Instance, device: &ash::Device, extensions: &DeviceExtensions) -> Self {
        let mut loaders = Self::default();

        if extensions.khr_swapchain {
            loaders.khr_swapchain = Some(Arc::new(khr::swapchain::Device::new(instance, device)));
        }

        if extensions.ray_tracing {
            loaders.khr_acceleration_structure = Some(Arc::new(khr::acceleration_structure::Device::new(instance, device)));
            loaders.khr_deferred_host_operations = Some(Arc::new(khr::deferred_host_operations::Device::new(instance, device)));
        }

        loaders
    }
}
