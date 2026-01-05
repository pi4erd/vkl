use ash::vk;

use crate::DefaultAllocator;

use crate::{Device, Instance, VklError, VklResult, alloc::{MemoryRequest, MemorySlice}};

pub struct Texture2d {
    allocator: DefaultAllocator,
    image: vk::Image,
    view: Option<vk::ImageView>,
    sampler: Option<vk::Sampler>,
    extent: vk::Extent2D,
    format: vk::Format,
    bound_memory: Option<MemorySlice>,
}

impl Texture2d {
    pub fn create(
        allocator: DefaultAllocator,
        device: &Device,
        extent: vk::Extent2D,
        tiling: vk::ImageTiling,
        format: vk::Format,
        image_usage: vk::ImageUsageFlags,
    ) -> VklResult<Self> {
        let image_info = vk::ImageCreateInfo::default()
            .extent(extent.into())
            .array_layers(1)
            .image_type(vk::ImageType::TYPE_2D)
            .mip_levels(1)
            .format(format) // TODO: Handle image formats
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED | image_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1); // TODO: Multisampling

        let image = unsafe {
            device.ffi().create_image(&image_info, None)
        }.map_err(|e| VklError::VulkanError(e))?;

        Ok(Self {
            allocator: allocator.clone(),
            extent,
            image,
            format,
            view: None,
            sampler: None,
            bound_memory: None,
        })
    }

    pub fn initialize_for_sampling(&mut self, device: &Device, aspect: vk::ImageAspectFlags) -> VklResult<()> {
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect)
            .base_array_layer(0)
            .base_mip_level(0)
            .layer_count(1)
            .level_count(1); // FIXME: Bunch of hard-coded values, will get in the way when multisample
        let view_info = vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(self.format)
            .subresource_range(subresource_range);

        let view = unsafe {
            device.ffi().create_image_view(&view_info, None)
        }.map_err(|e| VklError::VulkanError(e))?;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(device.properties.limits.max_sampler_anisotropy)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);
        let sampler = unsafe {
            device.ffi().create_sampler(&sampler_info, None)
        }.map_err(|e| VklError::VulkanError(e))?;

        self.view = Some(view);
        self.sampler = Some(sampler);

        Ok(())
    }

    pub fn create_depth_texture(instance: &Instance, allocator: DefaultAllocator, extent: vk::Extent2D) -> VklResult<Self> {
        let depth_format = Self::find_depth_format(instance)
            .ok_or(VklError::Custom(
                "No suitable depth format found".to_string().into())
            )?;
        let mut image = Self::create(
            allocator,
            instance.device(),
            extent,
            vk::ImageTiling::OPTIMAL,
            depth_format,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        )?;
        image.allocate_memory(vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryAllocateFlags::empty())?;
        image.initialize_for_sampling(instance.device(), vk::ImageAspectFlags::DEPTH)?;

        return Ok(image)
    }

    fn find_format(
        instance: &Instance,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags
    ) -> Option<vk::Format> {
        for candidate in candidates {
            let properties = unsafe { instance.instance.get_physical_device_format_properties(
                instance.device().physical_device,
                *candidate
            ) };

            match tiling {
                vk::ImageTiling::LINEAR if properties.linear_tiling_features & features == features => 
                    return Some(*candidate),
                vk::ImageTiling::OPTIMAL if properties.optimal_tiling_features & features == features =>
                    return Some(*candidate),
                vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT => todo!("Unsupported image tiling"),
                _ => {}
            }
        }

        None
    }

    fn find_depth_format(instance: &Instance) -> Option<vk::Format> {
        Self::find_format(
            instance,
            &[
                vk::Format::D32_SFLOAT, vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }

    pub fn view(&self) -> Option<vk::ImageView> {
        self.view
    }

    pub fn sampler(&self) -> Option<vk::Sampler> {
        self.sampler
    }

    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn allocate_memory(
        &mut self,
        properties: vk::MemoryPropertyFlags,
        alloc_flags: vk::MemoryAllocateFlags,
    ) -> VklResult<()> {
        let mut allocator = self.allocator
            .write()
            .expect("Failed to write lock allocator");

        let reqs = allocator.get_image_memory_requirements(self.image);

        let request = MemoryRequest {
            requirements: reqs,
            properties,
            alloc_flags,
        };
        let memory = allocator.get_memory(&request)?;

        allocator.bind_image_memory(self.image, &memory)?;
        self.bound_memory = Some(memory);

        Ok(())
    }

    #[cfg(feature = "image")]
    pub fn from_bytes(
        allocator: DefaultAllocator,
        device: &Device,
        bytes: &[u8],
        image_usage: vk::ImageUsageFlags,
    ) -> VklResult<Self> {
        use crate::Buffer;

        let image = image::load_from_memory(bytes)
            .map_err(|_| todo!("handle error"))?;

        let width = image.width();
        let height = image.height();
        
        let image_bytes = image.as_bytes();

        let extent = vk::Extent2D::default()
            .width(width)
            .height(height);
        let mut image = Self::create(
            allocator.clone(), device, extent,
            vk::ImageTiling::OPTIMAL, vk::Format::R8G8B8A8_SRGB,
            image_usage
        )?;

        image.allocate_memory(
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::empty(),
        )?;

        let mut staging_buffer = Buffer::create(
            allocator.clone(),
            device,
            &[super::QueueType::Graphics],
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::BufferCreateFlags::empty(),
            image_bytes.len() as u64,
        )?;
        staging_buffer.allocate_memory(
            device,
            vk::MemoryPropertyFlags::HOST_COHERENT |
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryAllocateFlags::empty(),
        )?;
        {
            let mut map = staging_buffer.map(device)?;

            map.write(image_bytes);
        }

        let transfer_buffer = device.allocate_command_buffer(
            super::QueueType::Graphics,
            vk::CommandBufferLevel::PRIMARY
        )?;

        {
            let encoder = device.create_command_encoder(
                *transfer_buffer,
                vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
            )?;

            encoder.transition_image_layout(
                image.image,
                image.format,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::NONE, vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER,
            );

            let subresource_layers = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1);
            let regions = [
                vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(subresource_layers)
                    .image_offset(vk::Offset3D::default())
                    .image_extent(image.extent.into())
            ];

            encoder.copy_buffer_to_image(
                &staging_buffer, image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions
            );

            encoder.transition_image_layout(
                image.image,
                image.format,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER,
            );
        }
        
        let submit_info = crate::SubmitInfo {
            command_buffers: &[&transfer_buffer],
            ..Default::default()
        };
        device.queue_submit(super::QueueType::Graphics, &[submit_info], None)?;
        device.queue_wait_idle(super::QueueType::Graphics)?;

        Ok(image)
    }

    pub fn get_description(&self) -> Option<vk::DescriptorImageInfo> {
        Some(
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.view()?)
                .sampler(self.sampler()?)
        )
    }
}

impl Drop for Texture2d {
    fn drop(&mut self) {
        let mut allocator = self.allocator
            .write()
            .expect("Failed to write-lock allocator");
        
        if let Some(mem) = self.bound_memory.as_ref() {
            allocator.uncommit_memory(mem);
        }

        unsafe {
            if let Some(sampler) = self.sampler {
                allocator.device().destroy_sampler(sampler, None);
            }

            if let Some(view) = self.view {
                allocator.device().destroy_image_view(view, None);
            }

            allocator.device().destroy_image(self.image, None);
        };
    }
}