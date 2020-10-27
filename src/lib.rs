pub(crate) mod common;
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk::*;
use common::*;
use nuklear::{
    Buffer as NkBuffer, Color, CommandBuffer, Context, ConvertConfig, DrawVertexLayoutAttribute,
    DrawVertexLayoutElements, DrawVertexLayoutFormat, Handle, Size, Vec2,
};
use shaderc::ShaderKind;
use std::str::from_utf8;
use std::sync::Arc;
use vk_mem::{
    Allocation, AllocationCreateFlags, AllocationCreateInfo, AllocationInfo, Allocator, MemoryUsage,
};

const TEXTURE_FORMAT: Format = Format::B8G8R8A8_UNORM;
const ORTHO: Ortho = [
    [2.0f32 / width as f32, 0.0f32, 0.0f32, 0.0f32],
    [0.0f32, -2.0f32 / height as f32, 0.0f32, 0.0f32],
    [0.0f32, 0.0f32, -1.0f32, 0.0f32],
    [-1.0f32, 1.0f32, 0.0f32, 1.0f32],
];

struct Vertex {
    position: [f32; 2],
    // "Position",
    uv: [f32; 2],
    // "TexCoord",
    color: [u8; 4], // "Color",
}

struct VkTexture {
    pub(crate) texture: VkImage,
    pub(crate) descriptor_data: VkDescriptorData,
}

struct VkImage {
    pub(crate) texture: Image,
    pub(crate) texture_view: ImageView,
    pub(crate) device_memory: DeviceMemory,
    pub(crate) allocation: Option<Allocation>,
    pub(crate) allocation_info: Option<AllocationInfo>,
}

struct VkDescriptorData {
    pub(crate) descriptor_layout: DescriptorSetLayout,
    pub(crate) descriptor_pool: DescriptorPool,
    pub(crate) descriptor_set: DescriptorSet,
}

type Ortho = [[f32; 4]; 4];

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
        mut allocator: Option<Allocator>,
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
            texture.texture,
        );
        let image_view = Self::create_image_view(device.as_ref(), texture.texture);
        texture.texture_view = image_view;
        let descriptor_set = Self::create_combined_sampler_descriptor_set(
            device.as_ref(),
            drawer.combined_sampler_layout,
            drawer.descriptor_pool,
            image_view,
            drawer.sampler,
        );

        unsafe {
            if let Some(mutable_allocator) = allocator.as_mut() {
                mutable_allocator.destroy_buffer(
                    staging_buffer.buffer,
                    staging_buffer.allocation.as_ref().unwrap(),
                );
            } else {
                device.unmap_memory(staging_buffer.device_memory);
                device.free_memory(staging_buffer.device_memory, None);
                device.destroy_buffer(staging_buffer.buffer, None);
            }
        }

        VkTexture {
            texture,
            descriptor_data: VkDescriptorData {
                descriptor_layout,
                descriptor_pool,
                descriptor_set,
            },
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
            device.begin_command_buffer(cmd_buffer, &begin_info);
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
                image,
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

            device.end_command_buffer(cmd_buffer);
            let cmd_buffers = [cmd_buffer];
            let submit_info = vec![SubmitInfo::builder()
                .command_buffers(&cmd_buffers[0..])
                .build()];
            device.queue_submit(graphics_queue, submit_info.as_slice(), fence);
            let fences = vec![fence];
            device.wait_for_fences(fences.as_slice(), true, u64::MAX);
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
            let image_view = device
                .create_image_view(&image_view_info, None)
                .expect("Failed to create image view.");
            image_view
        }
    }

    fn create_texture(
        device: &ash::Device,
        height: u32,
        width: u32,
        allocator: Option<Allocator>,
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
                .create_image(&image_info, &allocation_info)
                .expect("Failed to create texture for Nuklear.");
            VkImage {
                texture,
                device_memory: allocation_info.get_device_memory(),
                allocation: Some(allocation),
                allocation_info: Some(allocation_info),
                texture_view: ImageView::null(),
            }
        } else {
            unsafe {
                let texture = device
                    .create_image(&image_info, None)
                    .expect("Failed to create texture for Nuklear.");
                let memory_requirements = device.get_image_memory_requirements(texture);
                let allocation_info = MemoryAllocateInfo::builder()
                    .allocation_size(memory_requirements.size)
                    .memory_type_index(Self::find_memory_type_index(
                        instance,
                        physical_device,
                        memory_requirements.memory_type_bits,
                    ));
                let device_memory = device
                    .allocate_memory(&allocation_info, None)
                    .expect("Failed to allocate memory for staging buffer.");
                device.bind_image_memory(texture, device_memory, 0);
                VkImage {
                    texture,
                    device_memory,
                    allocation: None,
                    allocation_info: None,
                    texture_view: ImageView::null(),
                }
            }
        }
    }

    fn find_memory_type_index(
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
}

pub struct Drawer {
    pub color: Option<[f32; 4]>,
    pub(crate) combined_sampler_layout: DescriptorSetLayout,
    pub(crate) descriptor_pool: DescriptorPool,
    device: Arc<ash::Device>,
    command_buffer: NkBuffer,
    vertex_buffer_size: usize,
    index_buffer_size: usize,
    textures: Vec<VkTexture>,
    renderpass: RenderPass,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    sampler: Sampler,
    uniform_buffer: VkBuffer,
    uniform_buffer_layout: DescriptorSetLayout,
    uniform_buffer_descriptor_set: DescriptorSet,
    vertex_buffer: VkBuffer,
    index_buffer: VkBuffer,
    layout_elements: DrawVertexLayoutElements,
}

impl Drawer {
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Option<Allocator>,
        instance: &ash::Instance,
        physical_device: PhysicalDevice,
        command_buffer: NkBuffer,
        color: Option<[f32; 4]>,
        vertex_buffer_size: usize,
        index_buffer_size: usize,
        texture_count: usize,
    ) -> Self {
        let (vertex_shader, vs_info) =
            Self::create_shaders(device.as_ref(), "../shaders/vs.fx", ShaderKind::Vertex);
        let (fragment_shader, fs_info) =
            Self::create_shaders(device.as_ref(), "../shaders/fs.fx", ShaderKind::Fragment);
        let shader_infos = vec![vs_info, fs_info];

        let uniform_buffer = common::create_buffer(
            device.as_ref(),
            &ORTHO,
            std::mem::size_of::<Ortho>() as u64,
            allocator.clone(),
            instance,
            physical_device,
            BufferUsageFlags::UNIFORM_BUFFER,
            MemoryPropertyFlags::HOST_COHERENT | MemoryPropertyFlags::HOST_VISIBLE,
        );
        let uniform_descriptor_layout = common::create_descriptor_set_layout(
            device.as_ref(),
            0,
            DescriptorType::UNIFORM_BUFFER,
            ShaderStageFlags::VERTEX,
        );
        let texture_descriptor_layout = common::create_descriptor_set_layout(
            device.as_ref(),
            1,
            DescriptorType::COMBINED_IMAGE_SAMPLER,
            ShaderStageFlags::FRAGMENT,
        );
        let descriptor_pool = Self::create_descriptor_pool(device.as_ref(), texture_count as u32);
        let uniform_descriptor_set_layouts = [uniform_descriptor_layout];
        let uniform_descriptor_set = Self::create_uniform_descriptor_set(
            device.as_ref(),
            &uniform_descriptor_set_layouts[0..],
            descriptor_pool,
            uniform_buffer.buffer,
            std::mem::size_of::<Ortho>() as u64,
        );

        let descriptor_set_layouts = [uniform_descriptor_layout, texture_descriptor_layout];
        let renderpass = Self::create_renderpass(device.as_ref(), color.clone());
        let (pipeline, pipeline_layout) = Self::create_graphics_pipeline(
            device.as_ref(),
            renderpass,
            &descriptor_set_layouts[0..],
            shader_infos.as_slice(),
        );

        let vertex_data = vec![0_u8; vertex_buffer_size];
        let vertex_buffer = common::create_buffer(
            device.as_ref(),
            vertex_data.as_slice(),
            vertex_buffer_size as u64,
            allocator.clone(),
            instance,
            physical_device,
            BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        );

        let index_data = vec![0_u8; index_buffer_size];
        let index_buffer = common::create_buffer(
            device.as_ref(),
            index_data.as_slice(),
            index_buffer_size as u64,
            allocator.clone(),
            instance,
            physical_device,
            BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        );

        device.destroy_shader_module(vertex_shader, None);
        device.destroy_shader_module(fragment_shader, None);

        Drawer {
            color,
            uniform_buffer_layout: uniform_descriptor_layout,
            combined_sampler_layout: texture_descriptor_layout,
            descriptor_pool,
            device,
            command_buffer,
            vertex_buffer_size,
            index_buffer_size,
            textures: Vec::with_capacity(texture_count + 1),
            renderpass,
            pipeline,
            pipeline_layout,
            sampler,
            uniform_buffer,
            vertex_buffer,
            index_buffer,
            layout_elements: DrawVertexLayoutElements::new(&[
                (
                    DrawVertexLayoutAttribute::Position,
                    DrawVertexLayoutFormat::Float,
                    0,
                ),
                (
                    DrawVertexLayoutAttribute::TexCoord,
                    DrawVertexLayoutFormat::Float,
                    std::mem::size_of::<f32>() as Size * 2,
                ),
                (
                    DrawVertexLayoutAttribute::Color,
                    DrawVertexLayoutFormat::B8G8R8A8,
                    std::mem::size_of::<f32>() as Size * 4,
                ),
                (
                    DrawVertexLayoutAttribute::AttributeCount,
                    DrawVertexLayoutFormat::Count,
                    0,
                ),
            ]),
            uniform_buffer_descriptor_set: uniform_descriptor_set,
        }
    }

    pub fn add_texture(
        &mut self,
        queue: Queue,
        image: &[u8],
        width: u32,
        height: u32,
        instance: &ash::Instance,
        physical_device: PhysicalDevice,
        command_pool: CommandPool,
        allocator: Option<Allocator>,
    ) -> Handle {
        let device = self.device.clone();
        self.textures.push(VkTexture::new(
            device,
            queue,
            self,
            image,
            width,
            height,
            instance,
            physical_device,
            command_pool,
            allocator.clone(),
        ));
        Handle::from_id(self.textures.len() as i32)
    }

    pub fn draw(
        &mut self,
        ctx: &mut Context,
        cfg: &mut ConvertConfig,
        command_buffer: ash::vk::CommandBuffer,
        framebuffer: Framebuffer,
        extent: Extent2D,
        viewport: Viewport,
        width: u32,
        height: u32,
        scale: Vec2,
    ) {
        self.update(cfg);
        let clear_color = ClearColorValue {
            float32: self.color.unwrap_or([1.0, 2.0, 3.0, 1.0]),
        };

        let clear_values = vec![ClearValue { color: clear_color }];

        let renderpass_begin_info = RenderPassBeginInfo::builder()
            .render_pass(self.renderpass)
            .framebuffer(framebuffer)
            .render_area(
                Rect2D::builder()
                    .offset(Offset2D::builder().x(0).y(0).build())
                    .extent(extent)
                    .build(),
            )
            .clear_values(clear_values.as_slice());

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &renderpass_begin_info,
                SubpassContents::SECONDARY_COMMAND_BUFFERS,
            );
            let viewports = [viewport];
            self.device
                .cmd_set_viewport(command_buffer, 0, &viewports[0..]);
            self.device.cmd_bind_pipeline(
                command_buffer,
                PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            let vertex_buffers = [self.vertex_buffer.buffer];
            let offsets = [0];
            self.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &vertex_buffers[0..],
                &offsets[0..],
            );
            self.device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer.buffer,
                0,
                IndexType::UINT32,
            );
            let uniform_descriptor_sets = [self.uniform_buffer_descriptor_set];
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &uniform_descriptor_sets[0..],
                &[],
            );
            let mut start = 0;
            let mut end;
            for cmd in ctx.draw_command_iterator(&self.command_buffer) {
                if cmd.elem_count() < 1 {
                    continue;
                }
                let id = cmd.texture().id().expect("Failed to get texture id.");
                let res = self.find_resource(id);
                end = start + cmd.elem_count();
                let sampler_descriptor_sets = [self.com];
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    1,
                    &sampler_descriptor_sets[0..],
                    &[],
                );
                let scissors = [Rect2D::builder()
                    .offset(Offset2D::builder().x(0).y(0).build())
                    .extent(Extent2D::builder().height(height).width(width).build())];
                self.device
                    .cmd_set_scissor(command_buffer, 0, &scissors[0..]);
            }
        }
    }

    fn create_descriptor_pool(device: &ash::Device, texture_count: u32) -> DescriptorPool {
        let mut pool_sizes = vec![DescriptorPoolSize::builder()
            .descriptor_count(1)
            .ty(DescriptorType::UNIFORM_BUFFER)
            .build()];
        pool_sizes.push(
            DescriptorPoolSize::builder()
                .descriptor_count(texture_count)
                .ty(DescriptorType::COMBINED_IMAGE_SAMPLER)
                .build(),
        );
        let pool_info = DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes.as_slice())
            .max_sets(1)
            .build();
        unsafe {
            let descriptor_pool = device
                .create_descriptor_pool(&pool_info, None)
                .expect("Failed to create descriptor pool.");
            descriptor_pool
        }
    }

    fn create_uniform_descriptor_set(
        device: &ash::Device,
        descriptor_set_layout: &[DescriptorSetLayout],
        descriptor_pool: DescriptorPool,
        uniform_buffer: ash::vk::Buffer,
        buffer_range: DeviceSize,
    ) -> DescriptorSet {
        let allocate_info = DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(descriptor_set_layout);
        let descriptor_set: DescriptorSet;
        unsafe {
            let descriptor_sets = device
                .allocate_descriptor_sets(&allocate_info)
                .expect("Failed to create descriptor set.");
            descriptor_set = descriptor_sets[0];
        }
        let buffer_info = vec![DescriptorBufferInfo::builder()
            .buffer(uniform_buffer)
            .offset(0)
            .range(buffer_range)
            .build()];

        let write_descriptor = vec![WriteDescriptorSet::builder()
            .descriptor_type(DescriptorType::UNIFORM_BUFFER)
            .buffer_info(buffer_info.as_slice())
            .dst_array_element(0)
            .dst_binding(0)
            .dst_set(descriptor_set)
            .build()];

        unsafe {
            device.update_descriptor_sets(write_descriptor.as_slice(), &[]);
            descriptor_set
        }
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        renderpass: RenderPass,
        descriptor_set_layouts: &[DescriptorSetLayout],
        shader_stages: &[PipelineShaderStageCreateInfo],
    ) -> (Pipeline, PipelineLayout) {
        let mut vertex_attribute_descriptions = vec![];
        vertex_attribute_descriptions.push(
            VertexInputAttributeDescription::builder()
                .format(Format::R32G32_SFLOAT)
                .binding(0)
                .offset(memoffset::offset_of!(Vertex, position) as u32)
                .location(0)
                .build(),
        );
        vertex_attribute_descriptions.push(
            VertexInputAttributeDescription::builder()
                .format(Format::R32G32_SFLOAT)
                .binding(0)
                .offset(memoffset::offset_of!(Vertex, uv) as u32)
                .location(1)
                .build(),
        );
        vertex_attribute_descriptions.push(
            VertexInputAttributeDescription::builder()
                .format(Format::R32G32B32A32_SFLOAT)
                .binding(0)
                .offset(memoffset::offset_of!(Vertex, color) as u32)
                .location(2)
                .build(),
        );

        let vertex_binding_description = vec![VertexInputBindingDescription::builder()
            .binding(0)
            .input_rate(VertexInputRate::VERTEX)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .build()];

        let vi_info = PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(vertex_attribute_descriptions.as_slice())
            .vertex_binding_descriptions(vertex_binding_description.as_slice());

        let ia_info = PipelineInputAssemblyStateCreateInfo::builder()
            .primitive_restart_enable(false)
            .topology(PrimitiveTopology::TRIANGLE_LIST);

        let vp_info = PipelineViewportStateCreateInfo::builder()
            .scissor_count(1)
            .viewport_count(1)
            .build();

        let rs_info = PipelineRasterizationStateCreateInfo::builder()
            .cull_mode(CullModeFlags::NONE)
            .depth_bias_clamp(0.0)
            .depth_bias_constant_factor(0.0)
            .depth_bias_enable(false)
            .depth_bias_slope_factor(0.0)
            .depth_clamp_enable(false)
            .front_face(FrontFace::CLOCKWISE)
            .line_width(1.0)
            .polygon_mode(PolygonMode::FILL)
            .rasterizer_discard_enable(false);

        let color_attachment = vec![PipelineColorBlendAttachmentState::builder()
            .alpha_blend_op(BlendOp::ADD)
            .blend_enable(true)
            .color_blend_op(BlendOp::ADD)
            .color_write_mask(ColorComponentFlags::all())
            .dst_alpha_blend_factor(BlendFactor::ONE)
            .src_alpha_blend_factor(BlendFactor::ONE_MINUS_DST_ALPHA)
            .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
            .src_color_blend_factor(BlendFactor::SRC_ALPHA)
            .build()];

        let color_info = PipelineColorBlendStateCreateInfo::builder()
            .attachments(color_attachment.as_slice())
            .logic_op_enable(false);

        let dynamic_states = [DynamicState::SCISSOR, DynamicState::VIEWPORT];

        let dynamic_state_info =
            PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states[0..]);

        let pipeline_layout_info =
            PipelineLayoutCreateInfo::builder().set_layouts(descriptor_set_layouts);

        unsafe {
            let layout = device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create pipeline layout.");

            let pipeline_info = vec![GraphicsPipelineCreateInfo::builder()
                .render_pass(renderpass)
                .layout(layout)
                .base_pipeline_index(-1)
                .color_blend_state(&color_info)
                .dynamic_state(&dynamic_state_info)
                .input_assembly_state(&ia_info)
                .rasterization_state(&rs_info)
                .stages(shader_stages)
                .subpass(0)
                .vertex_input_state(&vi_info)
                .viewport_state(&vp_info)
                .build()];

            let pipeline = device
                .create_graphics_pipelines(PipelineCache::null(), pipeline_info.as_slice(), None)
                .expect("Failed to create graphics pipeline.");

            (pipeline[0], layout)
        }
    }

    fn create_renderpass(device: &ash::Device, color: Option<[f32; 4]>) -> RenderPass {
        let color_attachment = vec![AttachmentDescription::builder()
            .format(TEXTURE_FORMAT)
            .initial_layout(ImageLayout::UNDEFINED)
            .samples(SampleCountFlags::TYPE_1)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .load_op(if color.is_some() {
                AttachmentLoadOp::CLEAR
            } else {
                AttachmentLoadOp::LOAD
            })
            .final_layout(ImageLayout::PRESENT_SRC_KHR)
            .build()];

        let color_reference = vec![AttachmentReference::builder()
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .attachment(0)
            .build()];

        let subpass_description = vec![SubpassDescription::builder()
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
            .color_attachments(color_reference.as_slice())
            .build()];

        let mut subpass_dependencies = vec![SubpassDependency::builder()
            .dst_access_mask(AccessFlags::COLOR_ATTACHMENT_WRITE)
            .src_access_mask(AccessFlags::MEMORY_READ)
            .src_subpass(SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(PipelineStageFlags::FRAGMENT_SHADER)
            .dst_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dependency_flags(DependencyFlags::BY_REGION)
            .build()];

        subpass_dependencies.push(
            SubpassDependency::builder()
                .dst_access_mask(AccessFlags::MEMORY_READ)
                .src_access_mask(AccessFlags::COLOR_ATTACHMENT_WRITE)
                .src_subpass(0)
                .dst_subpass(SUBPASS_EXTERNAL)
                .src_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(PipelineStageFlags::FRAGMENT_SHADER)
                .dependency_flags(DependencyFlags::BY_REGION)
                .build(),
        );

        let renderpass_info = RenderPassCreateInfo::builder()
            .attachments(color_attachment.as_slice())
            .subpasses(subpass_description.as_slice())
            .dependencies(subpass_dependencies.as_slice());

        unsafe {
            let renderpass = device
                .create_render_pass(&renderpass_info, None)
                .expect("Failed to create renderpass for Nuklear.");
            renderpass
        }
    }

    fn create_sampler(device: &ash::Device) -> Sampler {
        let sampler_info = SamplerCreateInfo::builder()
            .unnormalized_coordinates(false)
            .mipmap_mode(SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(-100.0)
            .min_filter(Filter::LINEAR)
            .max_lod(100.0)
            .max_anisotropy(0.0)
            .mag_filter(Filter::LINEAR)
            .compare_op(CompareOp::ALWAYS)
            .compare_enable(false)
            .border_color(BorderColor::FLOAT_OPAQUE_WHITE)
            .anisotropy_enable(false)
            .address_mode_u(SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(SamplerAddressMode::CLAMP_TO_EDGE);

        unsafe {
            let sampler = device
                .create_sampler(&sampler_info, None)
                .expect("Failed to create sampler for Nuklear texture.");
            sampler
        }
    }

    fn create_shaders(
        device: &ash::Device,
        file_name: &str,
        shader_kind: ShaderKind,
    ) -> (ShaderModule, PipelineShaderStageCreateInfo) {
        let bytes = include_bytes!(file_name);
        let mut compiler = shaderc::Compiler::new().expect("Failed to initialize shader compiler.");
        let binary = compiler
            .compile_into_spirv(
                from_utf8(bytes).expect("Failed to read vs bytes into string."),
                shader_kind,
                file_name,
                "main",
                None,
            )
            .expect("Failed to compile vertex shader for Nuklear.");
        let shader_info = ShaderModuleCreateInfo::builder().code(binary.as_binary());
        unsafe {
            let shader = device
                .create_shader_module(&shader_info, None)
                .expect("Failed to create vertex shader module.");

            let entry_name =
                std::ffi::CString::new("main").expect("Failed to create entry name for shaders.");

            let shader_stage_info = PipelineShaderStageCreateInfo::builder()
                .name(entry_name.as_c_str())
                .module(shader)
                .stage(match shader_kind {
                    ShaderKind::Vertex => ShaderStageFlags::VERTEX,
                    ShaderKind::Fragment => ShaderStageFlags::FRAGMENT,
                    _ => ShaderStageFlags::empty(),
                })
                .build();

            (shader, shader_stage_info)
        }
    }

    fn find_resource(&self, id: i32) -> Option<&VkTexture> {
        if id > 0 && id as usize <= self.textures.len() {
            self.textures.get((id - 1) as usize)
        } else {
            None
        }
    }

    fn update(&mut self, cfg: &mut ConvertConfig) {
        cfg.set_vertex_layout(&self.layout_elements);
        cfg.set_vertex_size(std::mem::size_of::<Vertex>() as Size);
        {
            let vertex_buffer_mapped = self.vertex_buffer.mapped_memory;
            let vertex_buffer = unsafe {
                std::slice::from_raw_parts_mut(
                    vertex_buffer_mapped as *mut u8,
                    self.vertex_buffer_size / std::mem::size_of::<Vertex>(),
                )
            };
            let mut vertex_buffer = NkBuffer::with_fixed(vertex_buffer);

            let index_buffer_mapped = self.index_buffer.mapped_memory;
            let index_buffer = unsafe {
                std::slice::from_raw_parts_mut(
                    index_buffer_mapped as *mut u8,
                    self.index_buffer_size / std::mem::size_of::<u32>(),
                )
            };
            let mut index_buffer = NkBuffer::with_fixed(index_buffer);
            ctx.convert(
                &mut self.command_buffer,
                &mut vertex_buffer,
                &mut index_buffer,
                cfg,
            );
        }
    }
}
