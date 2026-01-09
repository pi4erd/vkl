pub mod debug_utils;
pub mod texture;
pub mod alloc;
pub mod piler;
pub mod exts;

pub use alloc::{
    Allocator,
    Buffer, BufferInfo, MemMap,
};

pub use texture::Texture2d;

pub type DefaultAllocator = Arc<RwLock<Allocator>>;

pub use piler::{
    ColorBlendAttachment, ColorBlendState, GraphicsPipelineInfo, Pipeline, PipelineColorState,
    PipelineDepthStencilState, PipelineHandle, PipelineLayoutHandle, PipelineManager,
    PipelineRasterizationState, PipelineRenderPassState, PipelineStage, PipelineVertexState,
    PipelineViewportState, PolygonMode, RenderPassHandle, ShaderModule, VertexAttribute,
    VertexBinding, DepthState, StencilState, ComputePipelineInfo,
};

pub use exts::{
    DeviceExtensions, DeviceFeatures
};

pub use debug_utils::{DebugCallback, DebugUtilsCallbackData, DebugUtilsMessageType, DebugUtilsMessageSeverity};

pub use ash::vk;

use exts::ExtensionLoaders;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    ffi::CStr,
    hash::{Hash, RandomState},
    ops::{Deref, Index, Range},
    rc::Rc,
    sync::{Arc, RwLock},
};

use ash::{ext, khr, prelude::*};

#[cfg(feature = "window")]
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

#[derive(Debug)]
pub enum VklError {
    VulkanError(vk::Result),
    VulkanLoadFailed(ash::LoadingError),
    MissingExtensions {
        extensions: Vec<String>,
    },
    MissingLayers {
        layers: Vec<String>,
    },
    NoSuitableDevice,
    NoQueueFamily(QueueType),
    BufferMapUnsupported,
    NoBoundMemory(String),
    OtherBufferError(String),
    Custom(Box<dyn std::error::Error>),
}

impl std::fmt::Display for VklError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VklError::VulkanError(err) => write!(f, "Vulkan Error: {}", err),
            VklError::VulkanLoadFailed(err) => write!(f, "Failed to load vulkan: {}", err),
            VklError::MissingExtensions { extensions } => {
                write!(f, "Missing required extensions ({}):\n", extensions.len())?;
                for ext in extensions {
                    write!(f, " - {}", ext)?;
                }
                Ok(())
            }
            VklError::MissingLayers { layers } => {
                write!(f, "Missing required layers ({}):\n", layers.len())?;
                for layer in layers {
                    write!(f, " - {}", layer)?;
                }
                Ok(())
            }
            VklError::NoQueueFamily(family_type) => write!(f, "No requested queue family found for queue type: {:?}", family_type),
            VklError::NoSuitableDevice => write!(f, "No suitable device found"),
            VklError::BufferMapUnsupported => write!(f, "Buffer created doesn't support map"),
            VklError::NoBoundMemory(message) => write!(f, "Buffer didn't have bound memory: {}", message),
            VklError::OtherBufferError(message) => write!(f, "Buffer error: {}", message),
            VklError::Custom(err) => write!(f, "Other error: {}", err),
        }
    }
}
impl std::error::Error for VklError {}

pub type VklResult<T> = Result<T, VklError>;

pub struct Entry(ash::Entry);

impl Entry {
    pub fn dynamic() -> VklResult<Self> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| VklError::VulkanLoadFailed(e))?;

        Ok(Self(entry))
    }

    #[cfg(feature = "linked")]
    pub fn linked() -> Self {
        Self(ash::Entry::linked())
    }
}

pub struct Instance {
    entry: Entry,
    instance: ash::Instance,
    messenger: Option<DebugMessenger>,
    surface: Option<Surface>,
    device: Option<Device>,
    debug_callback: Option<DebugCallback>,
    portability: bool,
}

impl Instance {
    pub fn new(
        entry: Entry,
        mut extensions: Vec<&CStr>,
        layers: Vec<&CStr>,
        portability: bool,
    ) -> VklResult<Self> {
        let available_extensions = unsafe { entry.0.enumerate_instance_extension_properties(None) }
            .map_err(|e| VklError::VulkanError(e))?;
        let available_layers = unsafe { entry.0.enumerate_instance_layer_properties() }
            .map_err(|e| VklError::VulkanError(e))?;

        if portability {
            extensions.push(khr::portability_enumeration::NAME);
        }

        if let Err(extensions) = Self::check_extensions(&extensions, &available_extensions) {
            log::error!("One or more required extensions wasn't found!");
            return Err(VklError::MissingExtensions { extensions });
        }

        if let Err(layers) = Self::check_layers(&layers, &available_layers) {
            log::error!("One or more required layers wasn't found!");
            return Err(VklError::MissingLayers { layers });
        }

        let enabled_extensions = extensions.iter().map(|e| e.as_ptr()).collect::<Vec<_>>();
        let enabled_layers = layers.iter().map(|l| l.as_ptr()).collect::<Vec<_>>();

        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_3)
            .application_name(c"My App")
            .engine_name(c"No engine");

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&enabled_extensions)
            .enabled_layer_names(&enabled_layers)
            .flags(if portability {
                vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
            } else {
                vk::InstanceCreateFlags::default()
            });

        let instance = unsafe { entry.0.create_instance(&instance_info, None) }
            .map_err(|e| VklError::VulkanError(e))?;

        log::info!("Created instance with {} extensions:", extensions.len());
        extensions.iter().for_each(|e| log::info!(" - {:?}", e));

        Ok(Self {
            entry,
            instance,
            messenger: None,
            surface: None,
            device: None,
            debug_callback: None,
            portability,
        })
    }

    pub fn create_allocator(&self) -> Arc<RefCell<alloc::Allocator>> {
        Arc::new(RefCell::new(alloc::Allocator::new(self, self.device())))
    }

    pub fn create_device(
        &mut self,
        mut extensions: DeviceExtensions,
        features: DeviceFeatures,
    ) -> VklResult<()> {
        if self.portability {
            extensions.khr_portability_subset = true;
        }

        let ext_vec = extensions.to_vec();

        let physical_device = Device::pick_physical_device(&self.instance, &ext_vec)
            .ok_or(VklError::NoSuitableDevice)?;

        let queue_families =
            QueueFamilies::new(&self.instance, self.surface.as_ref(), physical_device);

        let unique_queue_families = queue_families.unique_present();

        // Assume only a single queue for each family. We don't need any more
        let priorities = [1.0];
        let queues = unique_queue_families
            .iter()
            .map(|f| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(f.id)
                    .queue_priorities(&priorities)
            })
            .collect::<Vec<_>>();

        self.device = Some(Device::new(
            self,
            physical_device,
            extensions,
            features,
            queue_families,
            &queues,
        )?);

        Ok(())
    }

    #[cfg(feature = "window")]
    pub fn create_surface(&mut self, window: Arc<winit::window::Window>) -> VklResult<()> {
        self.surface = Some(
            Surface::from_window(self, window)?
        );

        Ok(())
    }

    unsafe extern "system" fn debug_callback(
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        type_flags: vk::DebugUtilsMessageTypeFlagsEXT,
        callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        user_data: *mut std::ffi::c_void,
    ) -> vk::Bool32 {
        let instance: &Self;
        if let Some(i) = unsafe { (user_data as *mut Self).as_ref() } {
            instance = i;
        } else {
            log::error!("BUG: user_data was NULL in debug callback.");
            return vk::FALSE
        }

        if let Some(callback) = instance.debug_callback {
            let callback_data = if let Some(callback_data) = unsafe { callback_data.as_ref() } {
                callback_data
            } else {
                log::error!("Callback data was NULL.");
                return vk::FALSE
            };

            let message = unsafe { CStr::from_ptr(callback_data.p_message) };
            let message = message.to_string_lossy();

            let callback_data = DebugUtilsCallbackData { message: &message };
            
            let severity = match severity {
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => DebugUtilsMessageSeverity::ERROR,
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => DebugUtilsMessageSeverity::WARNING,
                vk::DebugUtilsMessageSeverityFlagsEXT::INFO => DebugUtilsMessageSeverity::INFO,
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => DebugUtilsMessageSeverity::VERBOSE,
                _ => {
                    log::error!("Invalid severity: {:?}!", severity);
                    return vk::FALSE
                }
            };

            let message_type = match type_flags {
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => DebugUtilsMessageType::GENERAL,
                vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => DebugUtilsMessageType::VALIDATION,
                vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => DebugUtilsMessageType::PERFORMANCE,
                // TODO: Device address binding
                _ => {
                    log::error!("Invalid message type: {:?}!", type_flags);
                    return vk::FALSE
                }
            };

            return if callback(severity, message_type, &callback_data) { vk::TRUE } else { vk::FALSE }
        }

        vk::FALSE
    }

    pub fn create_messenger(
        &mut self,
        severity: DebugUtilsMessageSeverity,
        types: DebugUtilsMessageType,
        callback: DebugCallback,
    ) -> VklResult<()> {
        let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(severity.into())
            .message_type(types.into())
            .pfn_user_callback(Some(Self::debug_callback))
            .user_data(std::ptr::addr_of_mut!(*self) as *mut std::ffi::c_void);
        self.messenger = Some(DebugMessenger::new(self, &info)?);
        self.debug_callback = Some(callback);

        Ok(())
    }

    pub fn messenger(&self) -> &DebugMessenger {
        self.messenger.as_ref().expect("No debug messenger initialized")
    }

    pub fn surface(&self) -> &Surface {
        self.surface.as_ref().expect("No surface initialized")
    }

    pub fn device(&self) -> &Device {
        self.device.as_ref().expect("No device initialized")
    }

    pub fn device_mut(&mut self) -> &mut Device {
        self.device.as_mut().expect("No device initialized")
    }

    fn check_extensions(required: &[&CStr], available: &[vk::ExtensionProperties]) -> Result<(), Vec<String>> {
        log::debug!("Available instance extensions ({}):", available.len());
        available.iter().for_each(|l| {
            log::debug!(" - {:?}", l.extension_name_as_c_str().unwrap());
        });

        let mut not_found: Vec<String> = Vec::new();

        for &required_ext in required {
            let mut found = false;
            for available_ext in available {
                if available_ext.extension_name_as_c_str().unwrap() == required_ext {
                    found = true;
                    break;
                }
            }

            if !found {
                not_found.push(required_ext.to_string_lossy().to_string());
                log::error!("Unsupported extension: {:?}", required_ext);
            }
        }

        if !not_found.is_empty() {
            return Err(not_found)
        }

        Ok(())
    }

    fn check_layers(required: &[&CStr], available: &[vk::LayerProperties]) -> Result<(), Vec<String>> {
        log::debug!("Available instance layers ({}):", available.len());
        available.iter().for_each(|l| {
            log::debug!(" - {:?}", l.layer_name_as_c_str().unwrap());
        });

        let mut not_found: Vec<String> = Vec::new();

        for &required_layer in required {
            let mut found = false;
            for available_layer in available {
                if available_layer.layer_name_as_c_str().unwrap() == required_layer {
                    found = true;
                    break;
                }
            }

            if !found {
                not_found.push(required_layer.to_string_lossy().to_string());
                log::error!("Unsupported layer: {:?}", required_layer);
            }
        }

        if !not_found.is_empty() {
            return Err(not_found)
        }

        Ok(())
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        self.device = None;
        self.surface = None;
        self.messenger = None;

        unsafe {
            self.instance.destroy_instance(None);
        }
        log::debug!("Instance destroyed.");
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SwapchainInfo {
    pub width: u32,
    pub height: u32,
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    pub preferred_present_mode: vk::PresentModeKHR,
    pub preferred_format: vk::Format,
    pub preferred_color_space: vk::ColorSpaceKHR,
}

pub struct Swapchain {
    device: Arc<ash::Device>,
    loader: Arc<khr::swapchain::Device>,
    swapchain: vk::SwapchainKHR,

    pub present_mode: vk::PresentModeKHR,
    pub format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,

    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,

    framebuffers: Option<Rc<[vk::Framebuffer]>>,
}

impl Swapchain {
    fn new(
        device: &Device,
        surface: &Surface,
        info: &SwapchainInfo,
    ) -> VklResult<Self> {
        let loader = device.loaders.khr_swapchain.as_ref().ok_or(
            VklError::MissingExtensions { extensions: vec![
                vk::KHR_SWAPCHAIN_NAME.to_string_lossy().to_string()
            ] }
        )?.clone();

        let surface_formats = surface.get_surface_formats(device.physical_device)?;
        let present_modes = surface.get_surface_present_modes(device.physical_device)?;
        let caps = surface.get_surface_caps(device.physical_device)?;

        let extent = vk::Extent2D::default()
            .width(info.width)
            .height(info.height);
        let present_mode = Self::pick_present_mode(&present_modes, info.preferred_present_mode);
        let surface_format = Self::pick_surface_format(
            &surface_formats,
            info.preferred_format,
            info.preferred_color_space,
        );

        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 && image_count > caps.max_image_count {
            image_count = caps.max_image_count;
        }

        let swapchain = Self::create_swapchain(
            &loader,
            device,
            surface,
            image_count,
            extent,
            surface_format.color_space,
            surface_format.format,
            present_mode,
            &caps,
            info.composite_alpha,
        )?;

        let images = unsafe { loader.get_swapchain_images(swapchain) }
            .map_err(|e| VklError::VulkanError(e))?;
        let image_views = Self::create_image_views(device, &images, surface_format.format)?;

        Ok(Self {
            device: device.device.clone(),
            loader,
            swapchain,

            present_mode,
            format: surface_format,
            extent,

            images,
            image_views,
            framebuffers: None,
        })
    }

    pub fn create_framebuffers(
        &mut self,
        depth_attachment: Option<vk::ImageView>,
        render_pass: vk::RenderPass,
    ) -> VklResult<Rc<[vk::Framebuffer]>> {
        // TODO: Deprecate this and make users create fbs themselves
        let framebuffers = self
            .image_views
            .iter()
            .map(|&v| {
                let attachments = if let Some(depth) = depth_attachment {
                    [v, depth]
                } else {
                    [v, vk::ImageView::null()]
                };

                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .attachments(if depth_attachment.is_none() {
                        &attachments[0..1]
                    } else {
                        &attachments
                    })
                    .width(self.extent.width)
                    .height(self.extent.height)
                    .layers(1)
                    .render_pass(render_pass);
                unsafe { self.device.create_framebuffer(&framebuffer_info, None) }
            })
            .collect::<VkResult<Rc<[_]>>>()
            .map_err(|e| VklError::VulkanError(e))?;

        self.framebuffers = Some(framebuffers.clone());
        Ok(framebuffers)
    }

    // Both semaphore and fence are signal, not wait
    pub fn acquire_image(
        &self,
        sema: Option<&Semaphore>,
        fence: Option<&Fence>,
    ) -> VklResult<(u32, bool)> {
        unsafe {
            self.loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                if let Some(sema) = sema {
                    **sema
                } else {
                    vk::Semaphore::null()
                },
                if let Some(fence) = fence {
                    **fence
                } else {
                    vk::Fence::null()
                },
            ).map_err(|e| VklError::VulkanError(e))
        }
    }

    pub fn recreate(
        &mut self,
        device: &Device,
        surface: &Surface,
        config: &SwapchainInfo,
    ) -> VklResult<()> {
        unsafe { self.device.device_wait_idle() }
            .map_err(|e| VklError::VulkanError(e))?;

        self.destroy_framebuffers();
        self.destroy_image_views();
        self.destroy_swapchain();

        let surface_formats = surface.get_surface_formats(device.physical_device)?;
        let present_modes = surface.get_surface_present_modes(device.physical_device)?;
        let caps = surface.get_surface_caps(device.physical_device)?;

        let extent = vk::Extent2D::default()
            .width(config.width)
            .height(config.height);
        let present_mode = Self::pick_present_mode(&present_modes, config.preferred_present_mode);
        let surface_format = Self::pick_surface_format(
            &surface_formats,
            config.preferred_format,
            config.preferred_color_space,
        );

        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 && image_count > caps.max_image_count {
            image_count = caps.max_image_count;
        }

        let swapchain = Self::create_swapchain(
            &self.loader,
            device,
            surface,
            image_count,
            extent,
            surface_format.color_space,
            surface_format.format,
            present_mode,
            &caps,
            config.composite_alpha,
        )?;

        let images = unsafe { self.loader.get_swapchain_images(swapchain) }
            .map_err(|e| VklError::VulkanError(e))?;
        let image_views = Self::create_image_views(device, &images, surface_format.format)?;

        self.swapchain = swapchain;
        self.images = images;
        self.image_views = image_views;

        self.format = surface_format;
        self.present_mode = present_mode;
        self.extent = extent;

        Ok(())
    }

    pub fn get_image(&self, index: u32) -> Option<vk::Image> {
        self.images.get(index as usize).copied()
    }

    pub fn get_image_view(&self, index: u32) -> Option<vk::ImageView> {
        self.image_views.get(index as usize).copied()
    }

    pub fn get_image_count(&self) -> usize {
        self.images.len()
    }

    pub fn present(
        &self,
        device: &Device,
        image_index: u32,
        wait_semaphores: &[vk::Semaphore],
    ) -> VklResult<bool> {
        let images = [image_index];
        let swapchains = [self.swapchain];
        let present_info = vk::PresentInfoKHR::default()
            .image_indices(&images)
            .swapchains(&swapchains)
            .wait_semaphores(&wait_semaphores);
        unsafe {
            self.loader.queue_present(
                device.get_device_queue(QueueType::Present)
                    .ok_or(VklError::NoQueueFamily(QueueType::Present))?,
                &present_info,
            ).map_err(|e| VklError::VulkanError(e))
        }
    }

    fn create_swapchain(
        loader: &khr::swapchain::Device,
        device: &Device,
        surface: &Surface,
        image_count: u32,
        extent: vk::Extent2D,
        color_space: vk::ColorSpaceKHR,
        format: vk::Format,
        present_mode: vk::PresentModeKHR,
        caps: &vk::SurfaceCapabilitiesKHR,
        composite_alpha: vk::CompositeAlphaFlagsKHR,
    ) -> VklResult<vk::SwapchainKHR> {
        assert!(device.queue_families.is_complete_present());

        let graphics_qf = device.queue_families.get(QueueType::Graphics)
            .ok_or(VklError::NoQueueFamily(QueueType::Graphics))?;
        let present_qf = device.queue_families.get(QueueType::Present)
            .ok_or(VklError::NoQueueFamily(QueueType::Present))?;

        let image_sharing_mode;
        let queue_family_indices;
        if graphics_qf != present_qf {
            image_sharing_mode = vk::SharingMode::CONCURRENT;
            queue_family_indices = vec![graphics_qf.id, present_qf.id];
        } else {
            image_sharing_mode = vk::SharingMode::EXCLUSIVE;
            queue_family_indices = vec![];
        }

        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.surface)
            .min_image_count(image_count)
            .image_color_space(color_space)
            .image_format(format)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .present_mode(present_mode)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(caps.current_transform)
            .composite_alpha(composite_alpha)
            .clipped(true);
        let swapchain = unsafe { loader.create_swapchain(&swapchain_info, None) }
            .map_err(|e| VklError::VulkanError(e))?;

        log::debug!(
            "Created swapchain with extent {}x{} with {} images",
            extent.width,
            extent.height,
            image_count,
        );

        Ok(swapchain)
    }

    fn create_image_views(
        device: &Device,
        images: &[vk::Image],
        format: vk::Format,
    ) -> VklResult<Vec<vk::ImageView>> {
        images
            .iter()
            .map(|i| {
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .base_mip_level(0)
                    .layer_count(1)
                    .level_count(1);
                let image_view_info = vk::ImageViewCreateInfo::default()
                    .components(
                        vk::ComponentMapping::default()
                            .a(vk::ComponentSwizzle::IDENTITY)
                            .r(vk::ComponentSwizzle::IDENTITY)
                            .g(vk::ComponentSwizzle::IDENTITY)
                            .b(vk::ComponentSwizzle::IDENTITY),
                    )
                    .format(format)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .image(*i)
                    .subresource_range(subresource_range);
                unsafe { device.ffi().create_image_view(&image_view_info, None) }
            })
            .collect::<VkResult<Vec<_>>>()
            .map_err(|e| VklError::VulkanError(e))
    }

    fn destroy_framebuffers(&self) {
        if let Some(fbs) = self.framebuffers.as_ref() {
            fbs.iter().for_each(|&f| {
                unsafe { self.device.destroy_framebuffer(f, None) };
            });
        }
    }

    fn destroy_swapchain(&self) {
        unsafe { self.loader.destroy_swapchain(self.swapchain, None) };
    }

    fn destroy_image_views(&mut self) {
        self.image_views.iter().for_each(|&v| unsafe {
            self.device.destroy_image_view(v, None);
        });
        self.image_views.clear();
    }

    #[cfg(feature = "window")]
    fn pick_extent_from_window(window: &Window, caps: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if caps.current_extent.width != u32::MAX && caps.current_extent.height != u32::MAX {
            return caps.current_extent;
        } else {
            let size = window.inner_size();

            let mut actual_extent = vk::Extent2D {
                width: size.width,
                height: size.height,
            };

            actual_extent.width = actual_extent
                .width
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width);

            actual_extent.height = actual_extent
                .height
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height);

            return actual_extent;
        }
    }

    fn pick_present_mode(
        present_modes: &[vk::PresentModeKHR],
        preferred: vk::PresentModeKHR,
    ) -> vk::PresentModeKHR {
        *present_modes
            .iter()
            .find(|&&p| p == preferred)
            .unwrap_or(&vk::PresentModeKHR::FIFO)
    }

    fn pick_surface_format(
        formats: &[vk::SurfaceFormatKHR],
        format: vk::Format,
        color_space: vk::ColorSpaceKHR,
    ) -> vk::SurfaceFormatKHR {
        *formats
            .iter()
            .find(|f| f.format == format && f.color_space == color_space)
            .unwrap_or(&formats[0])
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe { self.device.device_wait_idle().expect("Device wait failed") };

        self.destroy_framebuffers();
        self.destroy_image_views();
        self.destroy_swapchain();

        log::debug!("Destroyed swapchain")
    }
}

pub struct Surface {
    loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
}

impl Surface {
    #[cfg(feature = "window")]
    fn from_window(instance: &Instance, window: Arc<winit::window::Window>) -> VklResult<Self> {
        let loader = khr::surface::Instance::new(&instance.entry.0, &instance.instance);
        let surface = unsafe {
            ash_window::create_surface(
                &instance.entry.0,
                &instance.instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
        }.map_err(|e| VklError::VulkanError(e))?;

        Ok(Self {
            loader,
            surface,
        })
    }

    pub fn get_surface_caps(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> VklResult<vk::SurfaceCapabilitiesKHR> {
        unsafe {
            self.loader
                .get_physical_device_surface_capabilities(physical_device, self.surface)
                .map_err(|e| VklError::VulkanError(e))
        }
    }

    pub fn get_surface_formats(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> VklResult<Vec<vk::SurfaceFormatKHR>> {
        unsafe {
            self.loader
                .get_physical_device_surface_formats(physical_device, self.surface)
                .map_err(|e| VklError::VulkanError(e))
        }
    }

    pub fn get_surface_present_modes(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> VklResult<Vec<vk::PresentModeKHR>> {
        unsafe {
            self.loader
                .get_physical_device_surface_present_modes(physical_device, self.surface)
                .map_err(|e| VklError::VulkanError(e))
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_surface(self.surface, None) };
    }
}

pub struct DebugMessenger {
    loader: ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl DebugMessenger {
    pub fn new(instance: &Instance, info: &vk::DebugUtilsMessengerCreateInfoEXT) -> VklResult<Self> {
        let loader = ext::debug_utils::Instance::new(&instance.entry.0, &instance.instance);

        let messenger = unsafe { loader.create_debug_utils_messenger(info, None) }
            .map_err(|e| VklError::VulkanError(e))?;

        log::debug!("Created messenger.");

        Ok(Self { loader, messenger })
    }
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.messenger, None);
        }
        log::debug!("Messenger destroyed.");
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QueueFamily {
    pub id: u32,
    pub properties: vk::QueueFamilyProperties,
}

impl PartialEq for QueueFamily {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for QueueFamily {}

impl Hash for QueueFamily {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QueueType {
    Graphics,
    Transfer,
    Compute,
    Present,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct QueueFamilies {
    pub graphics: Option<QueueFamily>,
    pub transfer: Option<QueueFamily>,
    pub compute: Option<QueueFamily>,
    pub present: Option<QueueFamily>,
}

impl QueueFamilies {
    pub fn new(
        instance: &ash::Instance,
        surface: Option<&Surface>,
        physical_device: vk::PhysicalDevice,
    ) -> Self {
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let mut me = Self::default();

        for (i, props) in queue_families.iter().enumerate() {
            if props.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                me.graphics = Some(QueueFamily {
                    id: i as u32,
                    properties: *props,
                });
            }

            if props.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                me.transfer = Some(QueueFamily {
                    id: i as u32,
                    properties: *props,
                });
            }

            if props.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                me.compute = Some(QueueFamily {
                    id: i as u32,
                    properties: *props,
                });
            }

            if let Some(surface) = surface {
                let surface_supported = unsafe {
                    surface.loader.get_physical_device_surface_support(
                        physical_device,
                        i as u32,
                        surface.surface,
                    )
                }
                .unwrap_or(false);

                if surface_supported {
                    me.present = Some(QueueFamily {
                        id: i as u32,
                        properties: *props,
                    });
                }
            }

            if me.is_complete() && surface.is_none() {
                return me;
            } else if me.is_complete_present() && surface.is_some() {
                return me;
            }
        }

        me
    }

    pub fn get(&self, queue_type: QueueType) -> Option<QueueFamily> {
        match queue_type {
            QueueType::Compute => self.compute,
            QueueType::Graphics => self.graphics,
            QueueType::Present => self.present,
            QueueType::Transfer => self.transfer,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.graphics.is_some() && self.transfer.is_some() && self.compute.is_some()
    }

    pub fn is_complete_present(&self) -> bool {
        self.graphics.is_some()
            && self.transfer.is_some()
            && self.compute.is_some()
            && self.present.is_some()
    }

    pub fn unique_present(&self) -> Box<[QueueFamily]> {
        let mut queue_families = HashSet::<QueueFamily, RandomState>::new();

        if let Some(graphics) = self.graphics {
            queue_families.insert(graphics);
        }
        
        if let Some(compute) = self.compute {
            queue_families.insert(compute);
        }
        
        if let Some(transfer) = self.transfer {
            queue_families.insert(transfer);
        }
        
        if let Some(present) = self.present {
            queue_families.insert(present);
        }

        queue_families
            .into_iter()
            .collect::<Box<_>>()
    }
}

pub struct Semaphore {
    device: Arc<ash::Device>,
    semaphore: vk::Semaphore,
}

impl Deref for Semaphore {
    type Target = vk::Semaphore;

    fn deref(&self) -> &Self::Target {
        &self.semaphore
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().expect("Device wait failed");
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

pub struct Fence {
    device: Arc<ash::Device>,
    fence: vk::Fence,
}

impl Deref for Fence {
    type Target = vk::Fence;

    fn deref(&self) -> &Self::Target {
        &self.fence
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().expect("Device wait failed");
            self.device.destroy_fence(self.fence, None);
        }
    }
}

pub struct CommandBuffer {
    device: Arc<ash::Device>,
    pool: vk::CommandPool,
    buf: vk::CommandBuffer,
}

impl Deref for CommandBuffer {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.buf
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.free_command_buffers(self.pool, &[self.buf]);
        }
    }
}

pub struct CommandBuffers {
    device: Arc<ash::Device>,
    pool: vk::CommandPool,
    bufs: Vec<vk::CommandBuffer>,
}

impl Index<usize> for CommandBuffers {
    type Output = vk::CommandBuffer;

    fn index(&self, index: usize) -> &Self::Output {
        &self.bufs[index]
    }
}

impl Index<u32> for CommandBuffers {
    type Output = vk::CommandBuffer;

    fn index(&self, index: u32) -> &Self::Output {
        &self.bufs[index as usize]
    }
}

impl Deref for CommandBuffers {
    type Target = [vk::CommandBuffer];

    fn deref(&self) -> &Self::Target {
        &self.bufs
    }
}

impl Drop for CommandBuffers {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().expect("Device wait failed");
            self.device.free_command_buffers(self.pool, &self.bufs);
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DescriptorPoolSize {
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
}

impl Into<vk::DescriptorPoolSize> for DescriptorPoolSize {
    fn into(self) -> vk::DescriptorPoolSize {
        vk::DescriptorPoolSize::default()
            .descriptor_count(self.descriptor_count)
            .ty(self.descriptor_type)
    }
}

pub struct DescriptorPool {
    device: Arc<ash::Device>,
    pool: vk::DescriptorPool,
}

impl DescriptorPool {
    pub fn new(device: &Device, pool_sizes: &[DescriptorPoolSize], max_sets: u32, flags: vk::DescriptorPoolCreateFlags) -> VklResult<Self> {
        let pool_sizes = pool_sizes
            .iter()
            .map(|&p| p.into())
            .collect::<Vec<vk::DescriptorPoolSize>>();

        let info = vk::DescriptorPoolCreateInfo::default()
            .flags(flags)
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);
        
        let pool = unsafe {
            device.ffi()
                .create_descriptor_pool(&info, None)
        }.map_err(|e| VklError::VulkanError(e))?;
        
        Ok(Self {
            device: device.device.clone(),
            pool,
        })
    }

    pub fn write_descriptor_sets(&self, writes: &[vk::WriteDescriptorSet], copies: &[vk::CopyDescriptorSet]) {
        unsafe {
            self.device.update_descriptor_sets(writes, copies)
        }
    }

    pub fn allocate_descriptor_set(&self, set_layout: &DescriptorSetLayout) -> VklResult<vk::DescriptorSet> {
        let layouts = [set_layout.layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(
                &layouts
            );
        Ok(unsafe {
            self.device.allocate_descriptor_sets(&alloc_info)
                .map_err(|e| VklError::VulkanError(e))?
        }[0])
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DescriptorSetLayoutBinding {
    pub descriptor_count: u32,
    pub descriptor_type: vk::DescriptorType,
    pub stage_flags: vk::ShaderStageFlags,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DescriptorSetLayoutInfo<'a> {
    pub bindings: &'a [DescriptorSetLayoutBinding],
    pub flags: vk::DescriptorSetLayoutCreateFlags,
}

pub struct DescriptorSetLayout {
    device: Arc<ash::Device>,
    layout: vk::DescriptorSetLayout,
}

impl DescriptorSetLayout {
    pub fn new(device: &Device, info: &DescriptorSetLayoutInfo) -> VklResult<Self> {
        let bindings = info.bindings
            .iter()
            .enumerate()
            .map(|(i, b)|
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_count(b.descriptor_count)
                    .descriptor_type(b.descriptor_type)
                    .stage_flags(b.stage_flags)
            )
            .collect::<Vec<_>>();
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(info.flags);
        let layout = unsafe {
            device.ffi()
                .create_descriptor_set_layout(&set_layout_info, None)
        }.map_err(|e| VklError::VulkanError(e))?;

        Ok(Self {
            device: device.device.clone(),
            layout,
        })
    }

    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct SubmitInfo<'a> {
    pub command_buffers: &'a [&'a CommandBuffer],
    pub signal_semaphores: &'a [&'a Semaphore],
    pub wait_semaphores: &'a [&'a Semaphore],
    pub wait_dst_stage_mask: &'a [vk::PipelineStageFlags],
}

pub struct CommandEncoder<'d> {
    device: &'d Device,
    cmd_buffer: vk::CommandBuffer,
}

impl<'d> CommandEncoder<'d> {
    pub fn new(device: &'d Device, cmd_buffer: vk::CommandBuffer, flags: vk::CommandBufferUsageFlags) -> VklResult<Self> {
        let info = vk::CommandBufferBeginInfo::default()
            .flags(flags);
        unsafe { 
            device.ffi().reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| VklError::VulkanError(e))?;
            device.ffi().begin_command_buffer(cmd_buffer, &info)
                .map_err(|e| VklError::VulkanError(e))?;
        };

        Ok(Self {
            device,
            cmd_buffer,
        })
    }
}

impl CommandEncoder<'_> {
    // stops encoder
    pub fn finish(self) {
        drop(self)
    }

    pub fn begin_rendering(
        &'_ self,
        render_info: &vk::RenderingInfo,
    ) -> ActiveRenderPass<'_> {
        unsafe { self.device.ffi().cmd_begin_rendering(self.cmd_buffer, render_info) };
        ActiveRenderPass {
            device: self.device,
            encoder: self,
            dynamic: true,
        }
    }

    pub fn begin_compute_pass(&'_ self) -> ActiveComputePass<'_> {
        ActiveComputePass { device:  self.device, encoder: self }
    }

    fn has_stencil_component(depth_format: vk::Format) -> bool {
        return match depth_format {
            vk::Format::D24_UNORM_S8_UINT |
            vk::Format::D32_SFLOAT_S8_UINT => true,
            _ => false
        }
    }

    pub fn transition_image_layout(
        &self,
        image: vk::Image,
        format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
    ) {
        let aspect_mask = match new_layout {
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => vk::ImageAspectFlags::DEPTH,
            _ => vk::ImageAspectFlags::COLOR,
        } | if Self::has_stencil_component(format) {
            vk::ImageAspectFlags::STENCIL
        } else {
            vk::ImageAspectFlags::empty()
        };
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_array_layer(0)
            .base_mip_level(0)
            .layer_count(1)
            .level_count(1);
        let image_barrier = vk::ImageMemoryBarrier::default()
            .image(image)
            .dst_access_mask(dst_access_mask)
            .src_access_mask(src_access_mask)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .subresource_range(subresource_range)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

        unsafe {
            self.device.ffi().cmd_pipeline_barrier(
                self.cmd_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_barrier],
            );
        }
    }

    pub fn begin_render_pass(
        &'_ self,
        info: &vk::RenderPassBeginInfo,
        contents: vk::SubpassContents,
    ) -> ActiveRenderPass<'_> {
        unsafe {
            self.device.ffi()
                .cmd_begin_render_pass(self.cmd_buffer, info, contents);
        }

        ActiveRenderPass {
            device: self.device,
            encoder: self,
            dynamic: false,
        }
    }

    pub fn copy_buffer(&self, src: &Buffer, dst: &Buffer, regions: &[vk::BufferCopy]) {
        unsafe {
            self.device.ffi()
                .cmd_copy_buffer(
                    self.cmd_buffer,
                    src.buffer(),
                    dst.buffer(),
                    regions,
                );
        }
    }

    pub fn copy_buffer_full(&self, src: &Buffer, dst: &Buffer) {
        assert!(dst.buffer_size() >= src.buffer_size());
        let region = vk::BufferCopy::default()
            .size(src.buffer_size())
            .dst_offset(0)
            .src_offset(0);
        self.copy_buffer(src, dst, &[region]);
    }

    pub fn copy_buffer_to_image(
        &self,
        src_buffer: &Buffer, image: vk::Image,
        image_layout: vk::ImageLayout, regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.device.ffi().cmd_copy_buffer_to_image(
                self.cmd_buffer,
                src_buffer.buffer(), image,
                image_layout, regions
            );
        }
    }

    // Extensions
    pub fn build_acceleration_structures(
        &self,
        build_infos: &[vk::AccelerationStructureBuildGeometryInfoKHR],
        build_ranges: &[&[vk::AccelerationStructureBuildRangeInfoKHR]],
    ) {
        let loader = self.device.loaders.khr_acceleration_structure.as_ref().expect("Extension not loaded");

        unsafe {
            loader.cmd_build_acceleration_structures(
                self.cmd_buffer,
                build_infos,
                build_ranges,
            )
        };
    }
}

impl Drop for CommandEncoder<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.ffi().end_command_buffer(self.cmd_buffer)
                .expect("Failed to end command buffer")
        }
    }
}

pub struct ActiveRenderPass<'e> {
    device: &'e Device,
    encoder: &'e CommandEncoder<'e>,
    dynamic: bool,
}

impl ActiveRenderPass<'_> {
    pub fn bind_vertex_buffers(
        &self,
        buffers: &[&Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        unsafe {
            self.device.ffi().cmd_bind_vertex_buffers(
                self.encoder.cmd_buffer,
                0,
                &buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>(),
                offsets,
            );
        }
    }

    pub fn bind_index_buffer(&self, buffer: &Buffer, offset: vk::DeviceAddress, index_type: vk::IndexType) {
        unsafe {
            self.device.ffi().cmd_bind_index_buffer(
                self.encoder.cmd_buffer,
                buffer.buffer(),
                offset,
                index_type
            );
        }
    }

    pub fn bind_descriptor_sets(
        &self,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &[vk::DescriptorSet],
    ) {
        unsafe {
            self.device.ffi().cmd_bind_descriptor_sets(
                self.encoder.cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0, descriptor_sets,
                &[],
            );
        }
    }

    pub fn set_viewports(&self, viewports: &[vk::Viewport]) {
        unsafe {
            self.device.ffi().cmd_set_viewport(self.encoder.cmd_buffer, 0, viewports);
        }
    }

    pub fn set_scissors(&self, scissors: &[vk::Rect2D]) {
        unsafe {
            self.device.ffi().cmd_set_scissor(self.encoder.cmd_buffer, 0, scissors);
        }
    }

    pub fn draw(
        &self,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) {
        unsafe {
            self.device.ffi().cmd_draw(
                self.encoder.cmd_buffer,
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            );
        }
    }

    pub fn draw_indexed(&self, vertex_offset: i32, indices: Range<u32>, instances: Range<u32>) {
        unsafe {
            self.device.ffi().cmd_draw_indexed(
                self.encoder.cmd_buffer,
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start, vertex_offset,
                instances.start
            );
        }
    }

    pub fn bind_pipeline(&self, pipeline: piler::Pipeline) {
        unsafe {
            let (pipeline, bind_point) = match pipeline {
                piler::Pipeline::Render(pipeline) => (pipeline, vk::PipelineBindPoint::GRAPHICS),
                piler::Pipeline::Compute(pipeline) => (pipeline, vk::PipelineBindPoint::COMPUTE),
                piler::Pipeline::RayTracing(pipeline) => {
                    (pipeline, vk::PipelineBindPoint::RAY_TRACING_KHR)
                }
            };

            self.device.ffi()
                .cmd_bind_pipeline(self.encoder.cmd_buffer, bind_point, pipeline);
        }
    }

    pub fn finish(self) {
        drop(self)
    }
}

impl Drop for ActiveRenderPass<'_> {
    fn drop(&mut self) {
        if self.dynamic {
            unsafe {
                self.device.ffi().cmd_end_rendering(self.encoder.cmd_buffer);
            }
        } else {
            unsafe {
                self.device.ffi().cmd_end_render_pass(self.encoder.cmd_buffer);
            }
        }
    }
}

pub struct ActiveComputePass<'e> {
    device: &'e Device,
    encoder: &'e CommandEncoder<'e>,
}

impl ActiveComputePass<'_> {
    pub fn dispatch(&self, groups: (u32, u32, u32)) {
        unsafe {
            self.device
                .ffi()
                .cmd_dispatch(
                    self.encoder.cmd_buffer,
                    groups.0, groups.1, groups.2,
                );
        }
    }

    pub fn bind_pipeline(&self, pipeline: piler::Pipeline) {
        let (pipeline, bind_point) = match pipeline {
            piler::Pipeline::Compute(pipeline) => (pipeline, vk::PipelineBindPoint::COMPUTE),
            _ => panic!("Cannot bind other pipelines except for COMPUTE to compute pass")
        };

        unsafe {
            self.device
                .ffi()
                .cmd_bind_pipeline(
                    self.encoder.cmd_buffer,
                    bind_point,
                    pipeline
                );
        }
    }

    pub fn bind_descriptor_sets(
        &self,
        pipeline_layout: vk::PipelineLayout,
        bind_point: vk::PipelineBindPoint,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
    ) {
        unsafe {
            self.device.ffi().cmd_bind_descriptor_sets(
                self.encoder.cmd_buffer,
                bind_point,
                pipeline_layout,
                first_set, descriptor_sets,
                &[],
            );
        }
    }

    pub fn finish(self) {
        drop(self);
    }
}

impl Drop for ActiveComputePass<'_> {
    fn drop(&mut self) {
        let memory_barrier = vk::MemoryBarrier::default()
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_access_mask(vk::AccessFlags::SHADER_WRITE);
        unsafe {
            self.device
                .ffi()
                .cmd_pipeline_barrier(
                    self.encoder.cmd_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[memory_barrier],
                    &[],
                    &[],
                );
        }
    }
}

pub struct Device {
    device: Arc<ash::Device>,
    physical_device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue_families: QueueFamilies,

    queues: HashMap<u32, vk::Queue>,
    command_pools: HashMap<u32, vk::CommandPool>,

    loaders: ExtensionLoaders,
}

impl Device {
    fn new(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        extensions: DeviceExtensions,
        features: DeviceFeatures,
        queue_families: QueueFamilies,
        queue_infos: &[vk::DeviceQueueCreateInfo],
    ) -> VklResult<Self> {
        let properties = unsafe {
            instance
                .instance
                .get_physical_device_properties(physical_device)
        };
        let memory_properties = unsafe {
            instance
                .instance
                .get_physical_device_memory_properties(physical_device)
        };

        // Extensions
        let exts_vec = extensions.to_vec();
        let enabled_extensions = exts_vec.iter().map(|e| e.as_ptr()).collect::<Vec<_>>();

        // Features
        let core_features = features.get_core_features();
        let mut feat_dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeatures::default()
            .dynamic_rendering(features.dynamic_rendering_enabled);
        let mut feat_ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::default()
            .ray_query(features.ray_tracing);
        let mut feat_acceleration_structure = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(features.ray_tracing)
            .acceleration_structure_host_commands(features.acceleration_structure_host_commands);
        let mut feat_buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
            .buffer_device_address(features.buffer_device_address);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(core_features)
            .push_next(&mut feat_dynamic_rendering)
            .push_next(&mut feat_ray_query)
            .push_next(&mut feat_acceleration_structure)
            .push_next(&mut feat_buffer_device_address);

        let device_info = vk::DeviceCreateInfo::default()
            .enabled_extension_names(&enabled_extensions)
            .push_next(&mut features)
            .queue_create_infos(queue_infos);

        let device = unsafe {
            instance
                .instance
                .create_device(physical_device, &device_info, None)
                .map_err(|e| VklError::VulkanError(e))?
        };

        log::info!("Created device with {} extensions:", exts_vec.len());
        exts_vec.iter().for_each(|e| log::info!(" - {:?}", e));

        let mut queues = HashMap::new();
        queue_infos.iter().for_each(|qi| {
            let queue = unsafe { device.get_device_queue(qi.queue_family_index, 0) };
            queues.insert(qi.queue_family_index, queue);
        });

        log::debug!("Acquired {} queues.", queues.len());

        let mut command_pools = HashMap::new();
        for qi in queue_infos.iter() {
            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(qi.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }
                .map_err(|e| VklError::VulkanError(e))?;

            command_pools.insert(qi.queue_family_index, command_pool);
        }

        let loaders = ExtensionLoaders::from_extensions(&instance.instance, &device, &extensions);

        Ok(Self {
            physical_device,
            device: Arc::new(device),
            properties,
            memory_properties,
            queue_families,

            queues,
            command_pools,

            loaders,
        })
    }

    pub fn get_device_queue(&self, queue_type: QueueType) -> Option<vk::Queue> {
        Some(self.queues[&self.queue_families.get(queue_type)?.id])
    }

    pub fn create_swapchain(
        &self,
        surface: &Surface,
        info: &SwapchainInfo,
    ) -> VklResult<Swapchain> {
        Swapchain::new(self, surface, info)
    }

    pub fn create_semaphore(&self) -> VklResult<Semaphore> {
        let sema_info = vk::SemaphoreCreateInfo::default();
        let semaphore = unsafe { self.device.create_semaphore(&sema_info, None) }
            .map_err(|e| VklError::VulkanError(e))?;

        Ok(Semaphore {
            device: self.device.clone(),
            semaphore,
        })
    }

    pub fn create_fence(&self, signaled: bool) -> VklResult<Fence> {
        let fence_info = vk::FenceCreateInfo::default().flags(if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::default()
        });
        let fence = unsafe { self.device.create_fence(&fence_info, None) }
            .map_err(|e| VklError::VulkanError(e))?;

        Ok(Fence {
            device: self.device.clone(),
            fence,
        })
    }

    pub fn create_command_encoder<'e>(
        &'e self,
        cmd_buffer: vk::CommandBuffer,
        flags: vk::CommandBufferUsageFlags
    ) -> VklResult<CommandEncoder<'e>> {
        CommandEncoder::new(self, cmd_buffer, flags)
    }

    pub fn get_command_pool(&self, queue_type: QueueType) -> Option<vk::CommandPool> {
        Some(self.command_pools[&self.queue_families.get(queue_type)?.id])
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        required_exts: &[&CStr],
    ) -> Option<vk::PhysicalDevice> {
        let physical_devices = unsafe { instance.enumerate_physical_devices() }.ok()?;

        let (mut picked_device, mut max_score) = (None, 0);

        for device in physical_devices {
            let properties = unsafe { instance.get_physical_device_properties(device) };
            let available_exts =
                unsafe { instance.enumerate_device_extension_properties(device) }.ok();

            if available_exts.is_none() {
                log::debug!(
                    "Device {:?} discarded: no extensions were available.",
                    properties.device_name_as_c_str().unwrap()
                );
                continue;
            }

            let available_exts = available_exts.unwrap();

            if !Self::check_extensions(required_exts, &available_exts) {
                log::debug!(
                    "Device {:?} discarded: one or more required extensions wasn't available",
                    properties.device_name_as_c_str().unwrap()
                );
                continue;
            }

            let score = match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 5,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 4,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 3,
                vk::PhysicalDeviceType::CPU => 2,
                _ => 1,
            };

            if max_score < score {
                picked_device = Some(device);
                max_score = score;
            }
        }

        if let Some(device) = picked_device {
            let properties = unsafe { instance.get_physical_device_properties(device) };

            log::debug!(
                "Picked physical device {:?}.",
                properties.device_name_as_c_str().unwrap()
            );
            _ = Self::print_device_extensions(instance, device);
        }

        picked_device
    }

    pub fn ffi(&self) -> &ash::Device {
        &self.device
    }

    fn print_device_extensions(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
    ) -> VklResult<()> {
        let available = unsafe { instance.enumerate_device_extension_properties(device) }
            .map_err(|e| VklError::VulkanError(e))?;
        log::debug!("Available device extensions ({}):", available.len());
        available.iter().for_each(|l| {
            log::debug!(" - {:?}", l.extension_name_as_c_str().unwrap());
        });

        Ok(())
    }

    fn check_extensions(required: &[&CStr], available: &[vk::ExtensionProperties]) -> bool {
        let mut all_found = true;
        for &required_ext in required {
            let mut found = false;
            for available_ext in available {
                if available_ext.extension_name_as_c_str().unwrap() == required_ext {
                    found = true;
                    break;
                }
            }

            if !found {
                all_found = false;
                log::error!("Unsupported device extension: {:?}", required_ext);
            }
        }

        all_found
    }
}

// API
impl Device {
    pub fn allocate_command_buffer(
        &self,
        queue_type: QueueType,
        level: vk::CommandBufferLevel,
    ) -> VklResult<CommandBuffer> {
        let pool = self
            .get_command_pool(queue_type)
            .ok_or(VklError::NoQueueFamily(queue_type))?;
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(level)
            .command_pool(pool)
            .command_buffer_count(1);

        let buf = unsafe {
            self.device.allocate_command_buffers(&alloc_info)
                .map_err(|e| VklError::VulkanError(e))?
        }[0];

        Ok(CommandBuffer {
            device: self.device.clone(),
            buf,
            pool,
        })
    }

    pub fn allocate_command_buffers(
        &self,
        queue_type: QueueType,
        level: vk::CommandBufferLevel,
        count: usize,
    ) -> VklResult<CommandBuffers> {
        let pool = self
            .get_command_pool(queue_type)
            .ok_or(VklError::NoQueueFamily(queue_type))?;
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(level)
            .command_pool(pool)
            .command_buffer_count(count as u32);

        let bufs = unsafe { self.device.allocate_command_buffers(&alloc_info) }
            .map_err(|e| VklError::VulkanError(e))?;

        Ok(CommandBuffers {
            device: self.device.clone(),
            bufs,
            pool,
        })
    }

    pub fn create_descriptor_pool(
        &self,
        max_sets: u32,
        pool_sizes: &[DescriptorPoolSize],
        flags: vk::DescriptorPoolCreateFlags
    ) -> VklResult<DescriptorPool> {
        DescriptorPool::new(self, pool_sizes, max_sets, flags)
    }

    pub fn create_descriptor_set_layout(
        &self,
        info: &DescriptorSetLayoutInfo,
    ) -> VklResult<DescriptorSetLayout> {
        DescriptorSetLayout::new(self, info)
    }

    pub fn create_pipeline_manager(&self) -> piler::PipelineManager {
        piler::PipelineManager::new(self)
    }

    pub fn free_command_buffers(&self, queue_type: QueueType, cmd_buffers: &[vk::CommandBuffer]) {
        unsafe {
            self.device
                .free_command_buffers(
                    self.get_command_pool(queue_type)
                        .expect("Command pool upon buffer destruction"),
                    cmd_buffers
                )
        };
    }

    pub fn wait_for_fences(&self, fences: &[&Fence]) -> VklResult<()> {
        unsafe {
            self.device.wait_for_fences(
                &fences.iter().map(|f| f.fence).collect::<Vec<_>>(),
                true,
                u64::MAX,
            ).map_err(|e| VklError::VulkanError(e))
        }
    }

    pub fn reset_fences(&self, fences: &[&Fence]) -> VklResult<()> {
        unsafe {
            self.device
                .reset_fences(&fences.iter().map(|f| f.fence).collect::<Vec<_>>())
                .map_err(|e| VklError::VulkanError(e))
        }
    }

    pub fn wait_idle(&self) -> VklResult<()> {
        unsafe { self.device.device_wait_idle() }
            .map_err(|e| VklError::VulkanError(e))
    }

    pub fn queue_submit(
        &self,
        queue_type: QueueType,
        submit_infos: &[SubmitInfo],
        fence: Option<&Fence>,
    ) -> VklResult<()> {
        // How many goddamn allocations are there??
        let command_buffers = submit_infos
            .iter()
            .map(|s|
                s.command_buffers
                    .iter()
                    .map(|c| c.buf)
                    .collect::<Vec<_>>()
            )
            .collect::<Vec<_>>();
        let wait_semaphores = submit_infos
            .iter()
            .map(|s|
                s.wait_semaphores
                    .iter()
                    .map(|s| s.semaphore)
                    .collect::<Vec<_>>()
            )
            .collect::<Vec<_>>();
        let signal_semaphores = submit_infos
            .iter()
            .map(|s|
                s.signal_semaphores
                    .iter()
                    .map(|s| s.semaphore)
                    .collect::<Vec<_>>()
            )
            .collect::<Vec<_>>();
        let submit_infos = submit_infos.iter()
            .enumerate()
            .map(|(i, s)| {
                let cmds = &command_buffers[i];
                let signal_semaphores = &signal_semaphores[i];
                let wait_semaphores = &wait_semaphores[i];

                vk::SubmitInfo::default()
                    .command_buffers(cmds)
                    .signal_semaphores(signal_semaphores)
                    .wait_dst_stage_mask(s.wait_dst_stage_mask)
                    .wait_semaphores(wait_semaphores)
            })
            .collect::<Vec<_>>();
        unsafe {
            self.device.queue_submit(
                self.get_device_queue(queue_type)
                    .ok_or(VklError::NoQueueFamily(queue_type))?,
                &submit_infos,
                if let Some(fence) = fence {
                    **fence
                } else {
                    vk::Fence::null()
                },
            ).map_err(|e| VklError::VulkanError(e))
        }
    }

    pub fn queue_wait_idle(&self, queue_type: QueueType) -> VklResult<()> {
        unsafe {
            self.device
                .queue_wait_idle(self.get_device_queue(queue_type).expect("No queue available to wait for"))
                .map_err(|e| VklError::VulkanError(e))
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.wait_idle().expect("Device wait failed");

            for (_, &pool) in self.command_pools.iter() {
                self.device.destroy_command_pool(pool, None);
            }

            self.device.destroy_device(None);
        }
        log::debug!("Destroyed device.");
    }
}
