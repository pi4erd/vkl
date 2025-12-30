compile_error!("Module unsupported");

use ash::{khr, prelude::VkResult, vk};
use std::sync::Arc;

use crate as vkl;

pub struct Blas {
    loader: Arc<khr::acceleration_structure::Device>,
    blas: vk::AccelerationStructureKHR,
    buffer: vkl::Buffer,
}

impl Blas {
    pub fn from_triangles(
        allocator: vkl::DefaultAllocator,
        device: &vkl::Device,
        queue_types: &[vkl::QueueType],
        build_flags: vk::BuildAccelerationStructureFlagsKHR,
        vertex_count: u32,
        index_count: u32,
        vertex_buffer: &vkl::Buffer,
        index_buffer: &vkl::Buffer,
    ) -> VkResult<Self> {
        let loader = device.loaders.khr_acceleration_structure
            .as_ref()
            .expect("Extension not loaded")
            .clone();

        let vertex_address = vertex_buffer.get_device_address();
        let index_address = index_buffer.get_device_address();

        let transform = vk::TransformMatrixKHR { matrix: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]};
        let transform_buffer = vkl::Buffer::new(
            allocator.clone(),
            device,
            &vkl::BufferInfo {
                data: &[transform],
                queue_types: &[vkl::QueueType::Graphics],
                buffer_usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR |
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                buffer_flags: vk::BufferCreateFlags::empty(),
                memory_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                alloc_flags: vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            },
        )?;
        let transform_address = transform_buffer.get_device_address();
        let geometry = vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::TRIANGLES,
            geometry: vk::AccelerationStructureGeometryDataKHR { triangles: vk::AccelerationStructureGeometryTrianglesDataKHR {
                vertex_format: vk::Format::R32G32B32_SFLOAT,
                vertex_stride: 3 * size_of::<f32>() as u64,
                vertex_data: vk::DeviceOrHostAddressConstKHR { device_address: vertex_address },
                max_vertex: vertex_count - 1,
                index_data: vk::DeviceOrHostAddressConstKHR { device_address: index_address },
                index_type: vk::IndexType::UINT32,
                transform_data: vk::DeviceOrHostAddressConstKHR { device_address: transform_address },
                ..Default::default()
            }},
            ..Default::default()
        };

        let geometries = [geometry];
        let mut geometry_build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
            ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            mode: vk::BuildAccelerationStructureModeKHR::BUILD,
            flags: build_flags,
            ..Default::default()
        }.geometries(&geometries);

        let primitive_count = index_count / 3;

        let mut build_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &geometry_build_info,
            &[primitive_count],
            &mut build_sizes
        ) };

        let as_size = build_sizes.acceleration_structure_size;
        let scratch_size = build_sizes.build_scratch_size;

        let mut blas_buffer = vkl::Buffer::create(
            allocator.clone(),
            device,
            queue_types,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR |
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::BufferCreateFlags::empty(),
            as_size,
        )?;
        blas_buffer.allocate_memory(
            device,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
        )?;

        let structure_info = vk::AccelerationStructureCreateInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .size(as_size)
            .buffer(blas_buffer.buffer());

        let blas = unsafe {
            loader.create_acceleration_structure(&structure_info, None)?
        };

        log::debug!("Created blas with {} triangles.", primitive_count);

        let mut scratch_buffer = vkl::Buffer::create(
            allocator.clone(),
            device,
            &[vkl::QueueType::Graphics],
            vk::BufferUsageFlags::STORAGE_BUFFER |
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::BufferCreateFlags::empty(),
            scratch_size,
        )?;
        scratch_buffer.allocate_memory(
            device,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
        )?;

        let scratch_address = scratch_buffer.get_device_address();
        geometry_build_info = geometry_build_info
            .dst_acceleration_structure(blas)
            .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_address });

        let build_cmd_buffer = device.allocate_command_buffer(
            vkl::QueueType::Graphics,
            vk::CommandBufferLevel::PRIMARY
        )?;

        let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .first_vertex(0)
            .primitive_count(primitive_count)
            .primitive_offset(0)
            .transform_offset(0);

        let encoder = device.create_command_encoder(
            *build_cmd_buffer,
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        )?;

        encoder.build_acceleration_structures(
            &[geometry_build_info],
            &[&[build_range_info]]
        );

        encoder.finish();

        let cmd_buffers = [*build_cmd_buffer];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cmd_buffers);
        device.queue_submit(vkl::QueueType::Graphics, &[submit_info], None)?;
        device.queue_wait_idle(vkl::QueueType::Graphics)?;

        log::debug!("Finished building blas ({:?}).", blas);

        return Ok(Self {
            loader,
            blas,
            buffer: blas_buffer,
        })
    }

    pub fn acceleration_structure(&self) -> vk::AccelerationStructureKHR {
        self.blas
    }

    pub fn get_device_address(&self) -> vk::DeviceAddress {
        unsafe {
            let addr_info = vk::AccelerationStructureDeviceAddressInfoKHR::default()
                .acceleration_structure(self.blas);
            self.loader.get_acceleration_structure_device_address(&addr_info)
        }
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_acceleration_structure(self.blas, None)
        };
    }
}

pub struct Tlas {
    loader: Arc<khr::acceleration_structure::Device>,
    tlas: vk::AccelerationStructureKHR,
    buffer: vkl::Buffer,
}

impl Tlas {
    pub fn new(
        allocator: vkl::DefaultAllocator,
        device: &vkl::Device,
        queue_types: &[vkl::QueueType],
        build_flags: vk::BuildAccelerationStructureFlagsKHR,
        blas: &Blas,
    ) -> VkResult<Self> {
        let loader = device.loaders.khr_acceleration_structure
            .as_ref()
            .expect("Extension not loaded")
            .clone();

        let blas_reference = vk::AccelerationStructureReferenceKHR { device_handle: blas.get_device_address() };
        let instances_array = [ // TODO: Add custom instance definitions
            vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR { matrix: [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                ]},
                instance_custom_index_and_mask: vk::Packed24_8::new(0, !0),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    (vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE).as_raw() as u8,
                ),
                acceleration_structure_reference: blas_reference,
            }
        ];

        let instances_buffer = vkl::Buffer::new(
            allocator.clone(),
            device,
            &vkl::BufferInfo {
                data: &instances_array,
                queue_types,
                buffer_flags: vk::BufferCreateFlags::empty(),
                buffer_usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR |
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                alloc_flags: vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            }
        )?;

        let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR { device_address: instances_buffer.get_device_address() });
        let geometry = vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::INSTANCES,
            geometry: vk::AccelerationStructureGeometryDataKHR { instances },
            ..Default::default()
        };

        let geometries = [geometry];
        let mut geometry_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(&geometries)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(build_flags);

        let primitive_count = instances_array.len() as u32;

        let mut build_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::HOST_OR_DEVICE,
            &geometry_build_info,
            &[primitive_count], // TODO
            &mut build_sizes
        ) };

        let as_size = build_sizes.acceleration_structure_size;
        let scratch_size = build_sizes.build_scratch_size;

        let mut tlas_buffer = vkl::Buffer::create(
            allocator.clone(),
            device,
            &[vkl::QueueType::Compute],
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR |
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::BufferCreateFlags::empty(),
            as_size,
        )?;
        tlas_buffer.allocate_memory(
            device,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
        )?;

        let tlas_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(tlas_buffer.buffer())
            .size(as_size)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);
        let tlas = unsafe {
            loader.create_acceleration_structure(&tlas_info, None)?
        };

        log::debug!("Created tlas with {} instances.", primitive_count);

        let mut scratch_buffer = vkl::Buffer::create(
            allocator.clone(),
            device,
            &[vkl::QueueType::Graphics],
            vk::BufferUsageFlags::STORAGE_BUFFER |
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::BufferCreateFlags::empty(),
            scratch_size,
        )?;
        scratch_buffer.allocate_memory(
            device,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
        )?;

        let scratch_address = scratch_buffer.get_device_address();
        geometry_build_info = geometry_build_info
            .dst_acceleration_structure(tlas)
            .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_address });

        let build_cmd_buffer = device.allocate_command_buffer(
            vkl::QueueType::Graphics,
            vk::CommandBufferLevel::PRIMARY
        )?;

        let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .first_vertex(0)
            .primitive_count(primitive_count)
            .primitive_offset(0)
            .transform_offset(0);

        let encoder = device.create_command_encoder(
            *build_cmd_buffer,
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        )?;

        encoder.build_acceleration_structures(
            &[geometry_build_info],
            &[&[build_range_info]]
        );

        encoder.finish();

        let cmd_buffers = [*build_cmd_buffer];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cmd_buffers);
        device.queue_submit(vkl::QueueType::Graphics, &[submit_info], None)?;
        device.queue_wait_idle(vkl::QueueType::Graphics)?;

        log::debug!("Finished building tlas ({:?}).", tlas);
        
        Ok(Self {
            loader,
            tlas,
            buffer: tlas_buffer,
        })
    }

    pub fn acceleration_structure(&self) -> vk::AccelerationStructureKHR {
        self.tlas
    }
}

impl Drop for Tlas {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_acceleration_structure(self.tlas, None);
        }
    }
}
