#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub(crate) mod common;
pub(crate) mod vk_texture;
pub(crate) use vk_texture::*;

use ash::version::DeviceV1_0;
use ash::vk::*;
use common::*;
use crossbeam::sync::ShardedLock;
use nuklear::{
    Buffer as NkBuffer, Context, ConvertConfig, DrawVertexLayoutAttribute,
    DrawVertexLayoutElements, DrawVertexLayoutFormat, Handle, Size, Vec2,
};
use shaderc::ShaderKind;
use std::str::from_utf8;
use std::sync::Arc;
use vk_mem::Allocator;

struct Vertex {
    // "Position"
    position: [f32; 2],
    // "TexCoord"
    uv: [f32; 2],
    // "Color"
    color: [f32; 4],
}

pub struct Drawer {
    pub color: Option<[f32; 4]>,
    pub(crate) combined_sampler_layout: DescriptorSetLayout,
    pub(crate) descriptor_pool: DescriptorPool,
    pub(crate) sampler: Sampler,
    device: Arc<ash::Device>,
    allocator: Option<Arc<ShardedLock<Allocator>>>,
    command_buffer: NkBuffer,
    vertex_buffer_size: usize,
    index_buffer_size: usize,
    textures: Vec<VkTexture>,
    renderpass: RenderPass,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
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
        allocator: Option<Arc<ShardedLock<Allocator>>>,
        instance: &ash::Instance,
        physical_device: PhysicalDevice,
        command_buffer: NkBuffer,
        color: Option<[f32; 4]>,
        vertex_buffer_size: usize,
        index_buffer_size: usize,
        texture_count: usize,
        color_format: Format,
        depth_format: Format,
        sample_count: SampleCountFlags,
    ) -> Self {
        let (vertex_shader, mut vs_info) =
            Self::create_shaders(device.as_ref(), "shaders/vs.vert", ShaderKind::Vertex);
        let (fragment_shader, mut fs_info) =
            Self::create_shaders(device.as_ref(), "shaders/fs.frag", ShaderKind::Fragment);
        let entry_name =
            std::ffi::CString::new("main").expect("Failed to create entry name for shaders.");
        vs_info.p_name = entry_name.as_ptr();
        fs_info.p_name = entry_name.as_ptr();
        let shader_infos = vec![vs_info, fs_info];

        let ortho_size = std::mem::size_of::<Ortho>();
        let empty_data = vec![0_u8; ortho_size];
        let uniform_buffer = common::create_buffer(
            device.as_ref(),
            empty_data.as_slice(),
            ortho_size as u64,
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
            0,
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
        let renderpass = Self::create_renderpass(
            device.as_ref(),
            color,
            color_format,
            depth_format,
            sample_count,
        );
        let (pipeline, pipeline_layout) = Self::create_graphics_pipeline(
            device.as_ref(),
            renderpass,
            &descriptor_set_layouts[0..],
            shader_infos.as_slice(),
            sample_count,
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

        let sampler = Self::create_sampler(device.as_ref());
        Drawer {
            color,
            uniform_buffer_layout: uniform_descriptor_layout,
            combined_sampler_layout: texture_descriptor_layout,
            sampler,
            descriptor_pool,
            device,
            allocator,
            command_buffer,
            vertex_buffer_size,
            index_buffer_size,
            textures: Vec::with_capacity(texture_count + 1),
            renderpass,
            pipeline,
            pipeline_layout,
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
        allocator: Option<Arc<ShardedLock<Allocator>>>,
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
            allocator,
        ));
        Handle::from_id(self.textures.len() as i32)
    }

    pub fn draw(
        &mut self,
        ctx: &mut Context,
        cfg: &mut ConvertConfig,
        command_buffer: ash::vk::CommandBuffer,
        viewport: Viewport,
        width: u32,
        height: u32,
        scale: Vec2,
    ) {
        self.update(ctx, cfg, width, height);

        unsafe {
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
            for cmd in ctx.draw_command_iterator(&self.command_buffer) {
                if cmd.elem_count() < 1 {
                    continue;
                }
                let id = cmd.texture().id().expect("Failed to get texture id.");
                let res = self.find_resource(id);
                let sampler_descriptor_set = res.expect("Failed to get resource.").descriptor_set;
                let sampler_descriptor_sets = [sampler_descriptor_set];
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
                    .extent(
                        Extent2D::builder()
                            .height((cmd.clip_rect().h * scale.y) as u32)
                            .width((cmd.clip_rect().w * scale.x) as u32)
                            .build(),
                    )
                    .build()];
                self.device
                    .cmd_set_scissor(command_buffer, 0, &scissors[0..]);
                self.device
                    .cmd_draw_indexed(command_buffer, cmd.elem_count(), 1, start, 0, 0);
                start += cmd.elem_count();
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
            .max_sets(1 + texture_count)
            .build();
        unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .expect("Failed to create descriptor pool.")
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
        sample_count: SampleCountFlags,
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

        let depth_info = PipelineDepthStencilStateCreateInfo::builder()
            .depth_bounds_test_enable(false)
            .depth_compare_op(CompareOp::LESS)
            .depth_test_enable(false)
            .depth_write_enable(true)
            .stencil_test_enable(false);

        let msaa_info = PipelineMultisampleStateCreateInfo::builder()
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            .min_sample_shading(0.25)
            .rasterization_samples(sample_count)
            .sample_shading_enable(false);

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
                .multisample_state(&msaa_info)
                .depth_stencil_state(&depth_info)
                .build()];

            let pipeline = device
                .create_graphics_pipelines(PipelineCache::null(), pipeline_info.as_slice(), None)
                .expect("Failed to create graphics pipeline.");

            (pipeline[0], layout)
        }
    }

    fn create_renderpass(
        device: &ash::Device,
        color: Option<[f32; 4]>,
        color_format: Format,
        depth_format: Format,
        sample_count: SampleCountFlags,
    ) -> RenderPass {
        let mut attachments = vec![AttachmentDescription::builder()
            .format(color_format)
            .initial_layout(ImageLayout::UNDEFINED)
            .samples(sample_count)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .load_op(if color.is_some() {
                AttachmentLoadOp::CLEAR
            } else {
                AttachmentLoadOp::LOAD
            })
            .final_layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];

        attachments.push(
            AttachmentDescription::builder()
                .format(depth_format)
                .initial_layout(ImageLayout::UNDEFINED)
                .samples(sample_count)
                .store_op(AttachmentStoreOp::STORE)
                .stencil_store_op(AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(AttachmentLoadOp::DONT_CARE)
                .load_op(AttachmentLoadOp::DONT_CARE)
                .final_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .build(),
        );

        attachments.push(
            AttachmentDescription::builder()
                .format(color_format)
                .initial_layout(ImageLayout::UNDEFINED)
                .samples(SampleCountFlags::TYPE_1)
                .store_op(AttachmentStoreOp::STORE)
                .stencil_store_op(AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(AttachmentLoadOp::DONT_CARE)
                .load_op(AttachmentLoadOp::DONT_CARE)
                .final_layout(ImageLayout::PRESENT_SRC_KHR)
                .build(),
        );

        let color_reference = vec![AttachmentReference::builder()
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .attachment(0)
            .build()];

        let depth_reference = AttachmentReference::builder()
            .attachment(1)
            .layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let resolve_reference = vec![AttachmentReference::builder()
            .attachment(2)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];

        let subpass_description = vec![SubpassDescription::builder()
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
            .color_attachments(color_reference.as_slice())
            .depth_stencil_attachment(&depth_reference)
            .resolve_attachments(resolve_reference.as_slice())
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
            .attachments(attachments.as_slice())
            .subpasses(subpass_description.as_slice())
            .dependencies(subpass_dependencies.as_slice());

        unsafe {
            device
                .create_render_pass(&renderpass_info, None)
                .expect("Failed to create renderpass for Nuklear.")
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
            device
                .create_sampler(&sampler_info, None)
                .expect("Failed to create sampler for Nuklear texture.")
        }
    }

    fn create_shaders(
        device: &ash::Device,
        file_name: &str,
        shader_kind: ShaderKind,
    ) -> (ShaderModule, PipelineShaderStageCreateInfo) {
        let bytes = std::fs::read(file_name).expect("Failed to read shader code.");
        let mut compiler = shaderc::Compiler::new().expect("Failed to initialize shader compiler.");
        let binary = compiler
            .compile_into_spirv(
                from_utf8(bytes.as_slice()).expect("Failed to read vs bytes into string."),
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

            let shader_stage_info = PipelineShaderStageCreateInfo::builder()
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

    fn update(&mut self, ctx: &mut Context, cfg: &mut ConvertConfig, width: u32, height: u32) {
        let ortho: Ortho = [
            [2.0f32 / width as f32, 0.0f32, 0.0f32, 0.0f32],
            [0.0f32, -2.0f32 / height as f32, 0.0f32, 0.0f32],
            [0.0f32, 0.0f32, -1.0f32, 0.0f32],
            [-1.0f32, 1.0f32, 0.0f32, 1.0f32],
        ];
        {
            let uniform_buffer_mapped = self.uniform_buffer.mapped_memory;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &ortho as *const _ as *const std::ffi::c_void,
                    uniform_buffer_mapped,
                    std::mem::size_of::<Ortho>(),
                );
            }
        }
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

impl Drop for Drawer {
    fn drop(&mut self) {
        let device = &self.device;
        for texture in self.textures.iter() {
            unsafe {
                device.destroy_image_view(texture.image.image_view, None);
            }
            if let Some(allocator) = self.allocator.as_ref() {
                let allocation = texture
                    .image
                    .allocation
                    .as_ref()
                    .expect("Failed to get allocation of image.");
                allocator
                    .read()
                    .expect("Failed to lock allocator.")
                    .destroy_image(texture.image.image, allocation)
                    .expect("Failed to destroy image.");
            } else {
                unsafe {
                    device.free_memory(texture.image.device_memory, None);
                    device.destroy_image(texture.image.image, None);
                }
            }
        }
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_render_pass(self.renderpass, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.combined_sampler_layout, None);
            device.destroy_descriptor_set_layout(self.uniform_buffer_layout, None);
            device.destroy_sampler(self.sampler, None);
        }
        if let Some(allocator) = self.allocator.as_ref() {
            let allocation = self
                .uniform_buffer
                .allocation
                .as_ref()
                .expect("Failed to get allocation for uniform buffer.");
            allocator
                .read()
                .expect("Failed to lock allocator.")
                .destroy_buffer(self.uniform_buffer.buffer, allocation)
                .expect("Failed to destroy uniform buffer.");
            let allocation = self
                .index_buffer
                .allocation
                .as_ref()
                .expect("Failed to get allocation for index buffer.");
            allocator
                .read()
                .expect("Failed to lock allocator.")
                .destroy_buffer(self.index_buffer.buffer, allocation)
                .expect("Failed to destroy index buffer.");
            let allocation = self
                .vertex_buffer
                .allocation
                .as_ref()
                .expect("Failed to get allocation for vertex buffer.");
            allocator
                .read()
                .expect("Failed to lock allocator.")
                .destroy_buffer(self.vertex_buffer.buffer, allocation)
                .expect("Failed to destroy vertex buffer.");
        } else {
            unsafe {
                device.unmap_memory(self.uniform_buffer.device_memory);
                device.free_memory(self.uniform_buffer.device_memory, None);
                device.destroy_buffer(self.uniform_buffer.buffer, None);
                device.unmap_memory(self.index_buffer.device_memory);
                device.free_memory(self.index_buffer.device_memory, None);
                device.destroy_buffer(self.index_buffer.buffer, None);
                device.unmap_memory(self.vertex_buffer.device_memory);
                device.free_memory(self.vertex_buffer.device_memory, None);
                device.destroy_buffer(self.vertex_buffer.buffer, None);
            }
        }
    }
}
