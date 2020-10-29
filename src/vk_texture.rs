#![allow(clippy::too_many_arguments, dead_code)]
use crate::{common, Drawer};
use ash::version::DeviceV1_0;
use ash::vk::*;
use crossbeam::sync::ShardedLock;
use std::sync::Arc;
use vk_mem::{
    Allocation, AllocationCreateFlags, AllocationCreateInfo, AllocationInfo, Allocator, MemoryUsage,
};

pub(crate) const TEXTURE_FORMAT: Format = Format::B8G8R8A8_UNORM;

pub(crate) struct VkTexture {
    pub(crate) image: VkImage,
    pub(crate) descriptor_set: DescriptorSet,
}

pub(crate) struct VkImage {
    pub(crate) image: Image,
    pub(crate) image_view: ImageView,
    pub(crate) device_memory: DeviceMemory,
    pub(crate) allocation: Option<Allocation>,
    pub(crate) allocation_info: Option<AllocationInfo>,
}

pub(crate) type Ortho = [[f32; 4]; 4];

impl VkTexture {
    pub fn new(
        device: Arc<ash::Device>,
        graphics_queue: Queue,
        drawer: &Drawer,
        image: &[u8],
        width: u32,
        height: u32,
        instance: &ash::Instance,
        physical_device: PhysicalDevice,
        command_pool: CommandPool,
        allocator: Option<Arc<ShardedLock<Allocator>>>,
    ) -> Self {
        let mut texture = Self::create_texture(
            device.as_ref(),
            height,
            width,
            allocator.clone(),
            instance,
            physical_device,
        );
        let staging_buffer = common::create_buffer(
            device.as_ref(),
            image,
            image.len() as u64,
            allocator.clone(),
            instance,
            physical_device,
            BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        );
        Self::copy_buffer_to_image(
            device.as_ref(),
            staging_buffer.buffer,
            command_pool,
            graphics_queue,
            width,
            height,
            texture.image,
        );
        let image_view = Self::create_image_view(device.as_ref(), texture.image);
        texture.image_view = image_view;
        let descriptor_set = Self::create_combined_sampler_descriptor_set(
            device.as_ref(),
            drawer.combined_sampler_layout,
            drawer.descriptor_pool,
            image_view,
            drawer.sampler,
        );

        unsafe {
            if let Some(allocator) = allocator.as_ref() {
                allocator
                    .read()
                    .expect("Failed to lock allocator.")
                    .destroy_buffer(
                        staging_buffer.buffer,
                        staging_buffer.allocation.as_ref().unwrap(),
                    )
                    .expect("Failed to destroy buffer.");
            } else {
                device.unmap_memory(staging_buffer.device_memory);
                device.free_memory(staging_buffer.device_memory, None);
                device.destroy_buffer(staging_buffer.buffer, None);
            }
        }

        VkTexture {
            image: texture,
            descriptor_set,
        }
    }

    fn copy_buffer_to_image(
        device: &ash::Device,
        buffer: Buffer,
        command_pool: CommandPool,
        graphics_queue: Queue,
        width: u32,
        height: u32,
        texture: Image,
    ) {
        let command_allocate_info = CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffer: ash::vk::CommandBuffer;
        unsafe {
            let cmd_buffers = device
                .allocate_command_buffers(&command_allocate_info)
                .expect("Failed to allocate command buffer.");
            cmd_buffer = cmd_buffers[0];
        }

        let fence_info = FenceCreateInfo::builder();
        let fence: Fence;
        unsafe {
            fence = device
                .create_fence(&fence_info, None)
                .expect("Failed to create fence.");
        }

        let begin_info =
            CommandBufferBeginInfo::builder().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .expect("Failed to begin command buffer.");
        }

        let copy_region = vec![BufferImageCopy::builder()
            .image_subresource(
                ImageSubresourceLayers::builder()
                    .mip_level(0)
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .base_array_layer(0)
                    .build(),
            )
            .image_extent(
                Extent3D::builder()
                    .width(width)
                    .height(height)
                    .depth(1)
                    .build(),
            )
            .build()];

        let mut barrier = ImageMemoryBarrier::builder()
            .image(texture)
            .dst_access_mask(AccessFlags::TRANSFER_WRITE)
            .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
            .new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
            .old_layout(ImageLayout::UNDEFINED)
            .src_queue_family_index(QUEUE_FAMILY_IGNORED)
            .subresource_range(
                ImageSubresourceRange::builder()
                    .base_array_layer(0)
                    .layer_count(1)
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .build(),
            )
            .build();

        unsafe {
            device.cmd_pipeline_barrier(
                cmd_buffer,
                PipelineStageFlags::TOP_OF_PIPE,
                PipelineStageFlags::TRANSFER,
                DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
            device.cmd_copy_buffer_to_image(
                cmd_buffer,
                buffer,
                texture,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                copy_region.as_slice(),
            );
            barrier.old_layout = ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = AccessFlags::SHADER_READ;
            device.cmd_pipeline_barrier(
                cmd_buffer,
                PipelineStageFlags::TRANSFER,
                PipelineStageFlags::FRAGMENT_SHADER,
                DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            device
                .end_command_buffer(cmd_buffer)
                .expect("Failed to end command buffer.");
            let cmd_buffers = [cmd_buffer];
            let submit_info = vec![SubmitInfo::builder()
                .command_buffers(&cmd_buffers[0..])
                .build()];
            device
                .queue_submit(graphics_queue, submit_info.as_slice(), fence)
                .expect("Failed to submit queue for execution.");
            let fences = vec![fence];
            device
                .wait_for_fences(fences.as_slice(), true, u64::MAX)
                .expect("Failed to wait for fence.");
            device.free_command_buffers(command_pool, &cmd_buffers[0..]);
            device.destroy_fence(fence, None);
        }
    }

    fn create_combined_sampler_descriptor_set(
        device: &ash::Device,
        descriptor_set_layout: DescriptorSetLayout,
        descriptor_pool: DescriptorPool,
        texture_view: ImageView,
        sampler: Sampler,
    ) -> DescriptorSet {
        unsafe {
            let layouts = vec![descriptor_set_layout];
            let allocate_info = DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(layouts.as_slice());
            let descriptor_sets = device
                .allocate_descriptor_sets(&allocate_info)
                .expect("Failed to create descriptor set.");

            let image_info = vec![DescriptorImageInfo::builder()
                .image_view(texture_view)
                .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .sampler(sampler)
                .build()];

            let write_descriptor = vec![WriteDescriptorSet::builder()
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(image_info.as_slice())
                .dst_array_element(0)
                .dst_binding(1)
                .dst_set(descriptor_sets[0])
                .build()];

            device.update_descriptor_sets(write_descriptor.as_slice(), &[]);

            descriptor_sets[0]
        }
    }

    fn create_image_view(device: &ash::Device, texture: Image) -> ImageView {
        let image_view_info = ImageViewCreateInfo::builder()
            .image(texture)
            .subresource_range(
                ImageSubresourceRange::builder()
                    .level_count(1)
                    .base_mip_level(0)
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .base_array_layer(0)
                    .build(),
            )
            .format(TEXTURE_FORMAT)
            .components(
                ComponentMapping::builder()
                    .r(ComponentSwizzle::R)
                    .g(ComponentSwizzle::G)
                    .b(ComponentSwizzle::B)
                    .a(ComponentSwizzle::A)
                    .build(),
            )
            .view_type(ImageViewType::TYPE_2D);

        unsafe {
            device
                .create_image_view(&image_view_info, None)
                .expect("Failed to create image view.")
        }
    }

    fn create_texture(
        device: &ash::Device,
        height: u32,
        width: u32,
        allocator: Option<Arc<ShardedLock<Allocator>>>,
        instance: &ash::Instance,
        physical_device: PhysicalDevice,
    ) -> VkImage {
        let image_info = ImageCreateInfo::builder()
            .format(TEXTURE_FORMAT)
            .samples(SampleCountFlags::TYPE_1)
            .initial_layout(ImageLayout::UNDEFINED)
            .mip_levels(1)
            .extent(
                Extent3D::builder()
                    .height(height)
                    .width(width)
                    .depth(1)
                    .build(),
            )
            .array_layers(1)
            .image_type(ImageType::TYPE_2D)
            .sharing_mode(SharingMode::EXCLUSIVE)
            .tiling(ImageTiling::OPTIMAL)
            .usage(ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST);

        if let Some(allocator) = allocator {
            let allocation_info = AllocationCreateInfo {
                usage: MemoryUsage::GpuOnly,
                flags: AllocationCreateFlags::NONE,
                required_flags: MemoryPropertyFlags::DEVICE_LOCAL,
                preferred_flags: MemoryPropertyFlags::empty(),
                memory_type_bits: 0,
                pool: None,
                user_data: None,
            };

            let (texture, allocation, allocation_info) = allocator
                .read()
                .expect("Failed to lock allocator.")
                .create_image(&image_info, &allocation_info)
                .expect("Failed to create texture for Nuklear.");
            VkImage {
                image: texture,
                device_memory: allocation_info.get_device_memory(),
                allocation: Some(allocation),
                allocation_info: Some(allocation_info),
                image_view: ImageView::null(),
            }
        } else {
            unsafe {
                let texture = device
                    .create_image(&image_info, None)
                    .expect("Failed to create texture for Nuklear.");
                let memory_requirements = device.get_image_memory_requirements(texture);
                let allocation_info = MemoryAllocateInfo::builder()
                    .allocation_size(memory_requirements.size)
                    .memory_type_index(common::find_memory_type_index(
                        instance,
                        physical_device,
                        memory_requirements.memory_type_bits,
                    ));
                let device_memory = device
                    .allocate_memory(&allocation_info, None)
                    .expect("Failed to allocate memory for staging buffer.");
                device
                    .bind_image_memory(texture, device_memory, 0)
                    .expect("Failed to bind image memory.");
                VkImage {
                    image: texture,
                    device_memory,
                    allocation: None,
                    allocation_info: None,
                    image_view: ImageView::null(),
                }
            }
        }
    }
}
