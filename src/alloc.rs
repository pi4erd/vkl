use std::{
    collections::{HashMap, HashSet}, hash::RandomState, os::raw::c_void, sync::Arc
};

use ash::vk;
use bytemuck::NoUninit;

use crate::{DefaultAllocator, VklError, VklResult};

use super::{Device, Instance};

pub type AllocHandle = u64;

pub const ALLOCATE_BLOCK_SIZE: u64 = 65536;

fn get_allocation_block_count(size: u64) -> u64 {
    let result_size = size / ALLOCATE_BLOCK_SIZE;
    result_size
        + if size % ALLOCATE_BLOCK_SIZE != 0 {
            1
        } else {
            0
        }
}

#[derive(Clone, Copy, Debug)]
pub struct MemorySlice {
    pub handle: AllocHandle,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub block_offset: u64,
}

impl MemorySlice {
    pub fn offset(&self) -> u64 {
        self.block_offset * ALLOCATE_BLOCK_SIZE
    }
}

#[derive(Clone, Debug)]
struct AllocatedMemory {
    memory: vk::DeviceMemory,
    memory_type: u32,
    allocation_flags: vk::MemoryAllocateFlags,
    size: vk::DeviceSize,

    bitvec: Vec<u8>,
}

impl AllocatedMemory {
    fn block_count(&self) -> u64 {
        get_allocation_block_count(self.size)
    }

    fn is_block_available(&self, block: u64) -> Option<bool> {
        if block >= self.block_count() { return None }

        let bit = block % 8;
        let bitvec_offset = block / 8;

        return Some(self.bitvec.get(bitvec_offset as usize)? & (1 << bit) == 0);
    }

    fn set_block_status(&mut self, block: u64, available: bool) {
        let bit = block % 8;
        let bitvec_offset = block / 8;

        if available {
            self.bitvec[bitvec_offset as usize] &= !(1 << bit);
        } else {
            self.bitvec[bitvec_offset as usize] |= 1 << bit;
        }
    }

    // returns offset to blocks that fit requirements, if found
    fn find_fit(&self, req: &MemoryRequest) -> Option<MemorySlice> {
        if self.size < req.requirements.size {
            return None;
        }

        let alignment_blocks = if req.requirements.alignment / ALLOCATE_BLOCK_SIZE == 0 {
            1
        } else {
            req.requirements.alignment / ALLOCATE_BLOCK_SIZE
        };

        let block_count = get_allocation_block_count(req.requirements.size);

        if block_count > self.bitvec.len() as u64 * 8 {
            return None;
        }

        let mut offset = 0;
        'find_loop: loop {
            if offset >= self.block_count() {
                return None;
            }

            // aligned find free spot
            for i in (offset..self.block_count()).step_by(alignment_blocks as usize) {
                if self.is_block_available(i)? {
                    offset = i;
                    break;
                }
            }

            // check if spot is contiguously free
            for i in offset..offset + block_count {
                if !self.is_block_available(i)? {
                    offset = ((offset + block_count) / alignment_blocks) * alignment_blocks + alignment_blocks;
                    continue 'find_loop;
                }
            }

            // sanity checks
            // assert because indicates problem with algorithm, not user issue
            assert_eq!(offset % alignment_blocks, 0);
            if offset + block_count > self.block_count() {
                return None
            }

            return Some(MemorySlice {
                handle: 0,
                memory: self.memory,
                size: req.requirements.size,
                block_offset: offset,
            });
        }
    }

    #[cfg(test)]
    fn find_fit_test() {
        let mut dummy_memory = Self {
            memory: vk::DeviceMemory::null(),
            allocation_flags: vk::MemoryAllocateFlags::empty(),
            memory_type: 0,
            size: ALLOCATE_BLOCK_SIZE * 16,
            bitvec: vec![0b00000000, 0b00000000],
        };

        let request = MemoryRequest {
            requirements: vk::MemoryRequirements {
                // Takes total 2 blocks
                size: ALLOCATE_BLOCK_SIZE * 2,
                alignment: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let fit = dummy_memory.find_fit(&request);

        assert_eq!(fit.unwrap().block_offset, 0);

        dummy_memory.bitvec[0] = 0b11101001;
        dummy_memory.bitvec[1] = 0b11111111;

        let fit = dummy_memory.find_fit(&request);

        assert_eq!(fit.unwrap().block_offset, 1);

        dummy_memory.bitvec[0] = 0b11101011;

        let fit = dummy_memory.find_fit(&request);

        assert!(fit.is_none());
    }

    fn commit(&mut self, mem: &MemorySlice) {
        let block_count = get_allocation_block_count(mem.size);

        assert!(
            mem.block_offset + block_count <= self.bitvec.len() as u64,
            "Memcommit failed. Block size incompatible ({} > {})",
            mem.block_offset + block_count,
            self.bitvec.len()
        );

        for i in mem.block_offset..mem.block_offset + block_count {
            self.set_block_status(i, false);
        }

        log::trace!(
            "Commited {} blocks of memory ({:?}) ({:x}..{:x})",
            block_count, mem.memory,
            mem.block_offset, mem.block_offset + block_count,
        );
    }

    fn uncommit(&mut self, mem: &MemorySlice) {
        let block_count = get_allocation_block_count(mem.size);

        assert!(mem.block_offset + block_count <= self.bitvec.len() as u64);

        for i in mem.block_offset..mem.block_offset + block_count {
            self.set_block_status(i, true);
        }

        log::trace!(
            "Uncommited {} blocks of memory ({:?}) ({:x}..{:x})",
            block_count, mem.memory,
            mem.block_offset, mem.block_offset + block_count,
        );
    }
}

#[test]
fn allocated_memory_tests() {
    AllocatedMemory::find_fit_test();
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryRequest {
    pub requirements: vk::MemoryRequirements,
    pub properties: vk::MemoryPropertyFlags,
    pub alloc_flags: vk::MemoryAllocateFlags,
}

#[derive(Clone, Copy, Default)]
pub struct BufferInfo<'d, T> {
    pub data: &'d [T],
    pub queue_types: &'d [super::QueueType],
    pub buffer_usage: vk::BufferUsageFlags,
    pub buffer_flags: vk::BufferCreateFlags,
    pub memory_properties: vk::MemoryPropertyFlags,
    pub alloc_flags: vk::MemoryAllocateFlags,
}

pub struct Allocator {
    device: Arc<ash::Device>,
    memory_properties: vk::PhysicalDeviceMemoryProperties,

    allocated_memory: HashMap<AllocHandle, AllocatedMemory>,

    last_alloc: AllocHandle,
}

impl Allocator {
    pub const DEFAULT_ALLOC_SIZE_BLOCKS: u64 = 20;

    pub fn new(instance: &Instance, device: &Device) -> Self {
        let memory_properties = unsafe {
            instance
                .instance
                .get_physical_device_memory_properties(device.physical_device)
        };

        log::debug!("Created allocator.");

        Self {
            device: device.device.clone(),
            memory_properties,
            allocated_memory: HashMap::new(),
            last_alloc: 0,
        }
    }

    pub fn device(&self) -> &ash::Device {
        self.device.as_ref()
    }

    pub fn get_image_memory_requirements(&self, image: vk::Image) -> vk::MemoryRequirements {
        assert_ne!(image, vk::Image::null());
        unsafe { self.device.get_image_memory_requirements(image) }
    }

    pub fn bind_image_memory(
        &self,
        image: vk::Image,
        memory: &MemorySlice,
    ) -> VklResult<()> {
        assert_ne!(image, vk::Image::null());
        assert_ne!(memory.memory, vk::DeviceMemory::null());

        unsafe {
            self.device.bind_image_memory(image, memory.memory, memory.offset())
        }.map_err(|e| VklError::VulkanError(e))
    }

    fn get_memory_type(&self, filter: u32, properties: vk::MemoryPropertyFlags) -> Option<u32> {
        for i in 0..self.memory_properties.memory_type_count {
            if (filter & (1 << i) != 0)
                && (self.memory_properties.memory_types[i as usize].property_flags & properties
                    == properties)
            {
                return Some(i);
            }
        }

        return None;
    }

    pub fn get_memory(&mut self, req: &MemoryRequest) -> VklResult<MemorySlice> {
        let requested_memory_type = self
            .get_memory_type(req.requirements.memory_type_bits, req.properties)
            .ok_or(vk::Result::ERROR_MEMORY_MAP_FAILED)
            .map_err(|e| VklError::VulkanError(e))?;

        let mut available_memory = None;

        for (&mem_idx, mem) in self.allocated_memory.iter() {
            if requested_memory_type != mem.memory_type {
                continue;
            }

            if req.alloc_flags & mem.allocation_flags != req.alloc_flags {
                continue;
            }

            let fit = mem.find_fit(req);

            if let Some(fit) = fit {
                available_memory = Some((mem_idx, fit));
                break;
            }
        }

        if available_memory.is_none() {
            let handle = self.request_memory(req)?;

            let memory = &self.allocated_memory[&handle];

            available_memory = Some((handle, memory.find_fit(&req).unwrap()));
        }

        let (mem_handle, mem) = available_memory.unwrap();

        let memory = self.allocated_memory.get_mut(&mem_handle).unwrap();

        memory.commit(&mem);

        log::trace!(
            "Commited memory for {} (0x{:x}..0x{:x})",
            mem_handle, mem.block_offset,
            mem.block_offset + get_allocation_block_count(mem.size)
        );

        return Ok(MemorySlice {
            handle: mem_handle,
            ..mem
        });
    }

    pub fn uncommit_memory(&mut self, mem: &MemorySlice) {
        if let Some(memory) = self.allocated_memory.get_mut(&mem.handle) {
            memory.uncommit(mem);
        } else {
            log::warn!("Trying to uncommit memory that wasn't allocated!");
        }
    }

    pub fn print_memory_usage(&self) {
        for (i, mem) in self.allocated_memory.iter() {
            let bitvec_str = mem.bitvec.iter()
                .fold(String::new(), |a, b| {
                    a + &format!("{:08b}", b).chars().rev().collect::<String>()
                });
            log::debug!("Memory {i}: {bitvec_str}");
        }
    }

    pub fn request_memory(&mut self, request: &MemoryRequest) -> VklResult<AllocHandle> {
        let memory_type = self
            .get_memory_type(request.requirements.memory_type_bits, request.properties)
            .ok_or(vk::Result::ERROR_MEMORY_MAP_FAILED)
            .map_err(|e| VklError::VulkanError(e))?;

        let requested_blocks = get_allocation_block_count(request.requirements.size);

        let allocated_blocks = Self::DEFAULT_ALLOC_SIZE_BLOCKS.max(requested_blocks);
        let allocation_size = allocated_blocks * ALLOCATE_BLOCK_SIZE;

        let mut alloc_flags = vk::MemoryAllocateFlagsInfo::default()
            .device_mask(0)
            .flags(request.alloc_flags);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(allocation_size)
            .memory_type_index(memory_type)
            .push_next(&mut alloc_flags);
        let memory = unsafe { self.device.allocate_memory(&alloc_info, None) }
            .map_err(|e| VklError::VulkanError(e))?;

        let alloc_handle = self.last_alloc;
        self.last_alloc += 1;

        self.allocated_memory.insert(
            alloc_handle,
            AllocatedMemory {
                memory,
                memory_type,
                allocation_flags: request.alloc_flags,
                size: allocation_size,
                bitvec: vec![0; allocated_blocks as usize],
            },
        );

        log::debug!(
            "Allocated new chunk of memory of size: {} bytes of type {} with allocation flags {:?}.",
            allocation_size,
            memory_type,
            alloc_flags.flags,
        );

        return Ok(alloc_handle);
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        for (_, mem) in self.allocated_memory.iter() {
            unsafe {
                self.device.free_memory(mem.memory, None);
            }
            log::trace!(
                "Cleared memory size: {} bytes of type {}.",
                mem.size,
                mem.memory_type
            );
        }

        self.allocated_memory.clear();

        log::debug!("Destroyed allocator.");
    }
}

pub struct MemMap<'w> {
    buffer: &'w Buffer,
    ptr: *mut c_void,
    size: u64,
}

impl MemMap<'_> {
    pub fn write<T: NoUninit>(&mut self, data: &[T]) {
        assert!(data.len() * size_of::<T>() <= self.size as usize, "Data out of bounds");
        
        let bytes: &[u8] = bytemuck::cast_slice(data);

        unsafe {
            std::ptr::copy::<u8>(
                bytes.as_ptr(),
                self.ptr as *mut u8,
                bytes.len()
            );
        }
    }

    pub fn read<T: bytemuck::AnyBitPattern>(&self) -> &[T] {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.ptr as *const u8,
                self.size as usize
            )
        };

        bytemuck::cast_slice(bytes)
    }
}

impl Drop for MemMap<'_> {
    fn drop(&mut self) {
        unsafe {
            self.buffer.allocator
                .read()
                .unwrap()
                .device
                .unmap_memory(
                    self.buffer.bound_memory.unwrap().memory
                );
        }
    }
}

pub struct Buffer {
    allocator: DefaultAllocator,
    buffer: vk::Buffer,
    buffer_size: vk::DeviceSize,
    bound_memory: Option<MemorySlice>,
    can_map: bool,
}

impl Buffer {
    pub fn create(
        allocator: DefaultAllocator,
        device: &Device,
        queue_types: &[super::QueueType],
        buffer_usage: vk::BufferUsageFlags,
        buffer_flags: vk::BufferCreateFlags,
        size: vk::DeviceSize,
    ) -> VklResult<Self> {
        let unique_queue_family_indices = queue_types
            .iter()
            .map(|&q| {
                let queue_family = device.queue_families.get(q);

                Ok(queue_family.ok_or(VklError::NoQueueFamily(q))?.id)
            })
            .collect::<VklResult<HashSet<u32, RandomState>>>()?;

        let queue_family_indices = unique_queue_family_indices.into_iter().collect::<Vec<_>>();

        let buffer_info = vk::BufferCreateInfo::default()
            .queue_family_indices(&queue_family_indices)
            .sharing_mode(if queue_family_indices.len() == 1 {
                vk::SharingMode::EXCLUSIVE
            } else {
                vk::SharingMode::CONCURRENT
            })
            .size(size)
            .usage(buffer_usage)
            .flags(buffer_flags);
        let buffer = unsafe { device.ffi().create_buffer(&buffer_info, None) }
            .map_err(|e| VklError::VulkanError(e))?;

        log::debug!("Created buffer with size {}", size);

        Ok(Self {
            allocator,
            buffer,
            bound_memory: None,
            buffer_size: size,
            can_map: false,
        })
    }

    pub fn get_device_address(&self) -> vk::DeviceAddress {
        let buf_addr = vk::BufferDeviceAddressInfo::default()
            .buffer(self.buffer);
        unsafe { self.allocator.read().unwrap().device.get_buffer_device_address(&buf_addr) }
    }

    pub fn buffer_size(&self) -> vk::DeviceSize {
        self.buffer_size
    }

    pub fn allocate_memory(
        &mut self,
        device: &Device,
        memory_properties: vk::MemoryPropertyFlags,
        alloc_flags: vk::MemoryAllocateFlags,
    ) -> VklResult<()> {
        let memory_requirements = unsafe {
            device
                .ffi()
                .get_buffer_memory_requirements(self.buffer)
        };

        let req = MemoryRequest {
            requirements: memory_requirements,
            properties: memory_properties,
            alloc_flags,
        };
        let memory = self.allocator.write().unwrap().get_memory(&req)?;

        unsafe {
            self.allocator.read().unwrap().device
                .bind_buffer_memory(self.buffer, memory.memory, memory.offset())
        }.map_err(|e| VklError::VulkanError(e))?;

        self.bound_memory = Some(memory);

        self.can_map = memory_properties.contains(vk::MemoryPropertyFlags::HOST_COHERENT) &&
            memory_properties.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);

        log::trace!("Bound memory ({:?}) offset {}.", memory.memory, memory.offset());

        return Ok(())
    }

    pub fn copy_buffer(&self, device: &Device, dst: &Buffer) -> VklResult<()> {
        let src_memory = self.bound_memory.ok_or(VklError::OtherBufferError(
            "Source buffer didn't have memory bound".to_string()
        ))?;

        let dst_memory = dst.bound_memory.ok_or(VklError::OtherBufferError(
            "Destination buffer didn't have memory bound".to_string()
        ))?;

        let transfer_cmd = device.allocate_command_buffer(
            super::QueueType::Transfer,
            vk::CommandBufferLevel::PRIMARY,
        )?;

        {
            let encoder = device.create_command_encoder(
                *transfer_cmd,
                vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
            )?;

            encoder.copy_buffer_full(self, dst);
        }

        let submit_info = super::SubmitInfo {
            command_buffers: &[&transfer_cmd],
            ..Default::default()
        };

        device.queue_submit(super::QueueType::Transfer, &[submit_info], None)?;
        device.queue_wait_idle(super::QueueType::Transfer)?;

        log::trace!(
            "Wrote {} bytes to memory ({:?}) offset {} from ({:?}) offset {}",
            self.buffer_size,
            dst_memory,
            dst_memory.offset(),
            src_memory,
            src_memory.offset(),
        );

        Ok(())
    }

    pub fn new<'d, T: NoUninit>(
        allocator: DefaultAllocator,
        device: &Device,
        info: &BufferInfo<'d, T>,
    ) -> VklResult<Self> {
        let size = (info.data.len() * size_of::<T>()) as u64;
        let mut buffer = Self::create(
            allocator.clone(),
            device,
            info.queue_types,
            info.buffer_usage | vk::BufferUsageFlags::TRANSFER_DST,
            info.buffer_flags,
            size,
        )?;

        buffer.allocate_memory(device, info.memory_properties, info.alloc_flags)?;

        buffer.write_data_staged(device, info.data)?;

        return Ok(buffer);
    }

    pub fn map<'w>(&'w self, device: &Device) -> VklResult<MemMap<'w>> {
        if !self.can_map {
            return Err(VklError::BufferMapUnsupported)
        }

        let memory = self.bound_memory.ok_or(VklError::NoBoundMemory(
            "Cannot map buffer without bound memory".to_string()
        ))?;

        let map = unsafe {
            device.ffi().map_memory(
                memory.memory,
                memory.offset(),
                memory.size,
                vk::MemoryMapFlags::empty(),
            )
        }.map_err(|e| VklError::VulkanError(e))?;

        Ok(MemMap {
            buffer: self,
            size: self.buffer_size,
            ptr: map,
        })
    }

    pub fn write_data_staged<T: NoUninit>(
        &self,
        device: &Device,
        data: &[T],
    ) -> VklResult<()> {
        let size = (size_of::<T>() * data.len()) as u64;
        let mut transfer_buffer = Self::create(
            self.allocator.clone(),
            device,
            &[super::QueueType::Transfer],
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::BufferCreateFlags::empty(),
            size,
        )?;

        transfer_buffer.allocate_memory(
            device, 
            vk::MemoryPropertyFlags::HOST_COHERENT |
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryAllocateFlags::empty(),
        )?;

        {
            let mut map = transfer_buffer.map(device)?;

            map.write(data);
        }

        transfer_buffer.copy_buffer(device, self)?;

        Ok(())
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn get_description(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let mut allocator = self.allocator.write().unwrap();

        if let Some(mem) = self.bound_memory.as_ref() {
            allocator.uncommit_memory(mem);
        }

        unsafe { allocator.device.destroy_buffer(self.buffer, None) };
    }
}
