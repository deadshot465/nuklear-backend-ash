#![allow(clippy::too_many_arguments, dead_code)]
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk::{
    Buffer, BufferCreateInfo, BufferUsageFlags, DescriptorSetLayout, DescriptorSetLayoutBinding,
    DescriptorSetLayoutCreateInfo, DescriptorType, DeviceMemory, MemoryAllocateInfo,
    MemoryMapFlags, MemoryPropertyFlags, PhysicalDevice, ShaderStageFlags, SharingMode,
};
use vk_mem::{
    Allocation, AllocationCreateFlags, AllocationCreateInfo, AllocationInfo, Allocator, MemoryUsage,
};

pub(crate) struct VkBuffer {
    pub(crate) buffer: Buffer,
    pub(crate) device_memory: DeviceMemory,
    pub(crate) mapped_memory: *mut std::ffi::c_void,
    pub(crate) allocation: Option<Allocation>,
    pub(crate) allocation_info: Option<AllocationInfo>,
}

pub(crate) fn create_buffer<'a, T>(
    device: &ash::Device,
    data: &[T],
    buffer_size: u64,
    allocator: Option<&'a Allocator>,
    instance: &ash::Instance,
    physical_device: PhysicalDevice,
    usage_flag: BufferUsageFlags,
    memory_properties: MemoryPropertyFlags,
) -> VkBuffer {
    let buffer_info = BufferCreateInfo::builder()
        .usage(usage_flag)
        .sharing_mode(SharingMode::EXCLUSIVE)
        .size(buffer_size);

    if let Some(allocator) = allocator {
        let allocation_info = AllocationCreateInfo {
            usage: match usage_flag {
                BufferUsageFlags::TRANSFER_SRC => MemoryUsage::CpuOnly,
                x if ((x & BufferUsageFlags::VERTEX_BUFFER) != BufferUsageFlags::empty()
                    || (x & BufferUsageFlags::INDEX_BUFFER) != BufferUsageFlags::empty())
                    && (x & BufferUsageFlags::TRANSFER_SRC) == BufferUsageFlags::empty() =>
                {
                    MemoryUsage::GpuOnly
                }
                _ => MemoryUsage::CpuToGpu,
            },
            flags: if (memory_properties & MemoryPropertyFlags::HOST_VISIBLE)
                == MemoryPropertyFlags::HOST_VISIBLE
                && (memory_properties & MemoryPropertyFlags::HOST_COHERENT)
                    == MemoryPropertyFlags::HOST_COHERENT
            {
                AllocationCreateFlags::MAPPED
            } else {
                AllocationCreateFlags::NONE
            },
            required_flags: memory_properties,
            preferred_flags: MemoryPropertyFlags::empty(),
            memory_type_bits: 0,
            pool: None,
            user_data: None,
        };

        let (buffer, allocation, allocation_info) = allocator
            .create_buffer(&buffer_info, &allocation_info)
            .expect("Failed to create staging buffer for Nuklear texture.");
        let device_memory = allocation_info.get_device_memory();
        let mapped = allocation_info.get_mapped_data();
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const std::ffi::c_void,
                mapped as *mut std::ffi::c_void,
                buffer_size as usize,
            );
        }
        VkBuffer {
            buffer,
            device_memory,
            allocation: Some(allocation),
            allocation_info: Some(allocation_info),
            mapped_memory: mapped as *mut std::ffi::c_void,
        }
    } else {
        unsafe {
            let buffer = device
                .create_buffer(&buffer_info, None)
                .expect("Failed to create staging buffer for Nuklear texture");
            let memory_requirements = device.get_buffer_memory_requirements(buffer);
            let allocation_info = MemoryAllocateInfo::builder()
                .allocation_size(memory_requirements.size)
                .memory_type_index(find_memory_type_index(
                    instance,
                    physical_device,
                    memory_requirements.memory_type_bits,
                ));
            let device_memory = device
                .allocate_memory(&allocation_info, None)
                .expect("Failed to allocate memory for staging buffer.");
            device
                .bind_buffer_memory(buffer, device_memory, 0)
                .expect("Failed to bind buffer memory.");
            let mapped = device
                .map_memory(device_memory, 0, buffer_size, MemoryMapFlags::empty())
                .expect("Failed to map memory for staging buffer.");
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const std::ffi::c_void,
                mapped,
                buffer_size as usize,
            );
            VkBuffer {
                buffer,
                device_memory,
                allocation: None,
                allocation_info: None,
                mapped_memory: mapped,
            }
        }
    }
}

pub(crate) fn create_descriptor_set_layout(
    device: &ash::Device,
    binding: u32,
    descriptor_type: DescriptorType,
    shader_stage: ShaderStageFlags,
) -> DescriptorSetLayout {
    let layout_binding = vec![DescriptorSetLayoutBinding::builder()
        .binding(binding)
        .descriptor_count(1)
        .descriptor_type(descriptor_type)
        .stage_flags(shader_stage)
        .build()];

    let layout_info = DescriptorSetLayoutCreateInfo::builder().bindings(layout_binding.as_slice());

    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .expect("Failed to create descriptor set layout.")
    }
}

pub(crate) fn find_memory_type_index(
    instance: &ash::Instance,
    physical_device: PhysicalDevice,
    memory_type: u32,
) -> u32 {
    unsafe {
        let properties = instance.get_physical_device_memory_properties(physical_device);
        let flags = MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT;
        for i in 0..properties.memory_type_count {
            if ((memory_type & (1 << i)) != 0)
                && ((properties.memory_types[i as usize].property_flags & flags) == flags)
            {
                return i;
            }
        }
    }
    0
}
