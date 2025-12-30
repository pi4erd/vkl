// Pipeline management module
use ash::vk;
use std::{collections::HashMap, ffi::CStr, io::Cursor, sync::Arc};

use crate::{VklError, VklResult};

pub type PipelineHandle = u64;
pub type PipelineLayoutHandle = u64;
pub type RenderPassHandle = u64;

#[derive(Clone, Copy)]
pub enum Pipeline {
    Render(vk::Pipeline),
    Compute(vk::Pipeline),
    RayTracing(vk::Pipeline),
}

impl Pipeline {
    pub fn pipeline(&self) -> vk::Pipeline {
        match self {
            Pipeline::Render(pipeline) => *pipeline,
            Pipeline::Compute(pipeline) => *pipeline,
            Pipeline::RayTracing(pipeline) => *pipeline,
        }
    }
}

pub struct ShaderModule {
    device: Arc<ash::Device>,
    module: vk::ShaderModule,
}

impl ShaderModule {
    fn new(device: Arc<ash::Device>, code: &[u32]) -> VklResult<Self> {
        let module_info = vk::ShaderModuleCreateInfo::default().code(code);
        let module = unsafe { device.create_shader_module(&module_info, None) }
            .map_err(|e| VklError::VulkanError(e))?;

        Ok(Self { device, module })
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.module, None);
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct ColorBlendState {
    pub alpha_blend_op: vk::BlendOp,
    pub color_blend_op: vk::BlendOp,
    pub dst_alpha_blend_factor: vk::BlendFactor,
    pub dst_color_blend_factor: vk::BlendFactor,
    pub src_alpha_blend_factor: vk::BlendFactor,
    pub src_color_blend_factor: vk::BlendFactor,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DepthState {
    pub depth_write: bool,
    pub depth_op: vk::CompareOp,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StencilState {
    // TODO
}

#[derive(Clone, Copy, Default)]
pub struct PipelineDepthStencilState {
    pub depth_state: DepthState,
    pub stencil_state: Option<StencilState>,
}

#[derive(Clone, Copy, Default)]
pub struct ColorBlendAttachment {
    pub color_write_mask: vk::ColorComponentFlags,
    pub blend_state: Option<ColorBlendState>,
    pub format: vk::Format,
}

#[derive(Clone, Copy, Default)]
pub struct PipelineColorState<'a> {
    pub attachments: &'a [ColorBlendAttachment],
    pub blend_constants: [f32; 4],
    pub logic_op: Option<vk::LogicOp>,
    pub flags: vk::PipelineColorBlendStateCreateFlags,
}

#[derive(Clone, Copy, Default)]
pub struct RasterizationDepthState {
    // TODO: Combine with PipelineDepthStencilState
}

#[derive(Clone, Copy, Debug)]
pub enum PolygonMode {
    Fill,
    Line(f32),
    Point(f32),
}

impl Default for PolygonMode {
    fn default() -> Self {
        Self::Fill
    }
}

#[derive(Clone, Copy, Default)]
pub struct PipelineRasterizationState {
    pub cull_mode: vk::CullModeFlags,
    pub depth_state: Option<RasterizationDepthState>,
    pub discard_enable: bool,
    pub front_face: vk::FrontFace,
    pub polygon_mode: PolygonMode,
}

#[derive(Clone, Copy, Default)]
pub struct VertexAttribute {
    pub format: vk::Format,
    pub offset: u32,
}

#[derive(Clone, Copy, Default)]
pub struct VertexBinding<'a> {
    pub stride: u32,
    pub input_rate: vk::VertexInputRate,
    pub attributes: &'a [VertexAttribute],
}

#[derive(Clone, Copy, Default)]
pub struct PipelineVertexState<'a> {
    pub topology: vk::PrimitiveTopology,
    pub bindings: &'a [VertexBinding<'a>],
    pub primitive_restart_enable: bool,
}

#[derive(Clone, Copy)]
pub struct PipelineStage<'a> {
    pub module: &'a ShaderModule,
    pub entrypoint: &'a CStr,
    // pub specialization: Option<&'a vk::SpecializationInfo<'a>>,
    pub stage: vk::ShaderStageFlags,
    pub flags: vk::PipelineShaderStageCreateFlags,
}

#[derive(Clone, Copy, Default)]
pub struct PipelineTesselationState {
    // TODO
}

#[derive(Clone, Copy, Default)]
pub struct PipelineMultisampleState {
    // TODO
}

#[derive(Clone, Copy)]
pub enum PipelineViewportState<'a> {
    Dynamic {
        viewport_count: u32,
        scissor_count: u32,
    },
    Static {
        viewports: &'a [vk::Viewport],
        scissors: &'a [vk::Rect2D],
    },
}

impl Default for PipelineViewportState<'_> {
    fn default() -> Self {
        Self::Dynamic {
            viewport_count: 1,
            scissor_count: 1,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct PipelineRenderPassState {
    pub render_pass: vk::RenderPass,
    pub subpass: u32,
}

#[derive(Clone, Default)]
pub struct GraphicsPipelineInfo<'a> {
    pub layout_ref: PipelineLayoutHandle,
    pub color_state: PipelineColorState<'a>,
    pub depth_stencil: Option<PipelineDepthStencilState>,
    pub dynamic_states: &'a [vk::DynamicState],
    pub vertex_state: PipelineVertexState<'a>,
    pub stages: &'a [PipelineStage<'a>],
    pub multisample: PipelineMultisampleState,
    pub rasterization: PipelineRasterizationState,
    pub viewport: PipelineViewportState<'a>,
    pub tesselation: Option<PipelineTesselationState>,
    pub render_pass: Option<PipelineRenderPassState>,
    pub flags: vk::PipelineCreateFlags,
}

#[derive(Clone)]
pub struct ComputePipelineInfo<'a> {
    pub layout_ref: PipelineLayoutHandle,
    pub stage: PipelineStage<'a>,
    pub flags: vk::PipelineCreateFlags,
}

pub struct PipelineManager {
    device: Arc<ash::Device>,

    pipelines: HashMap<PipelineHandle, Pipeline>,
    layouts: HashMap<PipelineLayoutHandle, vk::PipelineLayout>,
    render_passes: HashMap<RenderPassHandle, vk::RenderPass>,

    last_pipeline: PipelineHandle,
    last_layout: PipelineLayoutHandle,
    last_render_pass: RenderPassHandle,
}

impl PipelineManager {
    pub fn new(device: &super::Device) -> Self {
        Self {
            device: device.device.clone(),

            pipelines: HashMap::new(),
            layouts: HashMap::new(),
            render_passes: HashMap::new(),

            last_layout: 0,
            last_pipeline: 0,
            last_render_pass: 0,
        }
    }

    pub fn create_render_pass(
        &mut self,
        info: &vk::RenderPassCreateInfo,
    ) -> VklResult<RenderPassHandle> {
        let render_pass = unsafe { self.device.create_render_pass(info, None) }
            .map_err(|e| VklError::VulkanError(e))?;

        let handle = self.last_render_pass;
        self.last_render_pass += 1;

        self.render_passes.insert(handle, render_pass);

        Ok(handle)
    }

    pub fn create_shader_module(&self, code: &[u8]) -> VklResult<ShaderModule> {
        let mut reader = Cursor::new(code);
        let code = ash::util::read_spv(&mut reader)
            .map_err(|e| VklError::Custom(Box::new(e)))?;

        Ok(ShaderModule::new(self.device.clone(), &code)?)
    }

    pub fn create_pipeline_layout(
        &mut self,
        set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> VklResult<PipelineLayoutHandle> {
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(set_layouts)
            .push_constant_ranges(push_constant_ranges);
        let layout = unsafe { self.device.create_pipeline_layout(&info, None) }
            .map_err(|e| VklError::VulkanError(e))?;

        let handle = self.last_layout;
        self.last_layout += 1;

        self.layouts.insert(handle, layout);

        return Ok(handle);
    }

    pub fn get_render_pass(&self, handle: RenderPassHandle) -> vk::RenderPass {
        self.render_passes[&handle]
    }

    pub fn get_layout(&self, handle: PipelineLayoutHandle) -> vk::PipelineLayout {
        self.layouts[&handle]
    }

    pub fn get_pipeline(&self, handle: PipelineHandle) -> Pipeline {
        self.pipelines[&handle]
    }

    pub fn create_compute_pipelines(
        &mut self,
        infos: &[ComputePipelineInfo]
    ) -> Result<Vec<PipelineHandle>, (Vec<PipelineHandle>, vk::Result)> {
        let stages = infos
            .iter()
            .map(|info| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(info.stage.stage)
                    .module(info.stage.module.module)
                    .name(info.stage.entrypoint)
                    .flags(info.stage.flags)
            })
            .collect::<Vec<_>>();
        let infos = std::iter::zip(infos, stages)
            .map(|(info, stage)| {
                vk::ComputePipelineCreateInfo::default()
                    .layout(self.get_layout(info.layout_ref))
                    .stage(stage)
                    .flags(info.flags)
            })
            .collect::<Vec<_>>();
        
        let mut error: Option<vk::Result> = None;
        let pipelines = match unsafe {
            self.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &infos,
                None,
            )
        } {
            Ok(p) => p,
            Err((p, e)) => {
                error = Some(e);
                p
            },
        };

        let handles = pipelines
            .iter()
            .map(|p| self.append_pipeline(Pipeline::Compute(*p)))
            .collect::<Vec<_>>();

        if let Some(error) = error {
            return Err((handles, error))
        }

        return Ok(handles)
    }

    pub fn create_compute_pipeline(&mut self, info: &ComputePipelineInfo) -> VklResult<PipelineHandle> {
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(info.stage.stage)
            .module(info.stage.module.module)
            .name(info.stage.entrypoint)
            .flags(info.stage.flags);
        let info = vk::ComputePipelineCreateInfo::default()
            .layout(self.get_layout(info.layout_ref))
            .stage(stage)
            .flags(info.flags);

        let pipeline = match unsafe {
            self.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[info],
                None
            )
        } {
            Ok(p) => Ok(p[0]),
            Err((_, e)) => {
                Err(e)
            }
        }.map_err(|e| VklError::VulkanError(e))?;

        let handle = self.append_pipeline(Pipeline::Compute(pipeline));
        
        Ok(handle)
    }

    pub fn create_graphics_pipeline(
        &mut self,
        info: &GraphicsPipelineInfo,
    ) -> VklResult<PipelineHandle> {
        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .flags(info.flags);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .depth_attachment_format(vk::Format::default())
            .view_mask(0)
            .stencil_attachment_format(vk::Format::default());

        // Fill color state
        let mut color_blend_state = vk::PipelineColorBlendStateCreateInfo::default();

        let color_state = info.color_state;

        let color_attachment_formats = color_state.attachments.iter()
            .map(|a| a.format)
            .collect::<Vec<_>>();
        rendering_info = rendering_info.color_attachment_formats(&color_attachment_formats);

        let attachments = color_state
            .attachments
            .iter()
            .map(|a| {
                let mut attachment = vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(a.color_write_mask)
                    .blend_enable(false);

                if let Some(blend) = a.blend_state {
                    attachment = attachment
                        .blend_enable(true)
                        .alpha_blend_op(blend.alpha_blend_op)
                        .color_blend_op(blend.color_blend_op)
                        .dst_alpha_blend_factor(blend.dst_alpha_blend_factor)
                        .src_alpha_blend_factor(blend.src_alpha_blend_factor)
                        .dst_color_blend_factor(blend.dst_color_blend_factor)
                        .dst_color_blend_factor(blend.src_color_blend_factor)
                }

                attachment
            })
            .collect::<Vec<_>>();

        color_blend_state = color_blend_state
            .blend_constants(color_state.blend_constants)
            .attachments(&attachments)
            .logic_op_enable(false)
            .flags(color_state.flags);

        if let Some(logic_op) = color_state.logic_op {
            color_blend_state = color_blend_state.logic_op_enable(true).logic_op(logic_op);
        }

        // Fill depth/stencil state
        let mut depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default();
        
        if let Some(depth_state) = info.depth_stencil {
            depth_stencil_state = depth_stencil_state
                .depth_test_enable(true)
                .depth_write_enable(depth_state.depth_state.depth_write)
                .depth_compare_op(depth_state.depth_state.depth_op);

            if let Some(_stencil_state) = depth_state.stencil_state {
                todo!()
            }
        }

        // Fill dynamic state
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(info.dynamic_states);

        // Fill vertex state
        let vertex_state = info.vertex_state;

        let mut vertex_bindings = Vec::new();
        let mut vertex_attributes = Vec::new();

        for (i, binding) in vertex_state.bindings.iter().enumerate() {
            for (j, attribute) in binding.attributes.iter().enumerate() {
                let attribute_info = vk::VertexInputAttributeDescription::default()
                    .binding(i as u32)
                    .location(j as u32)
                    .format(attribute.format)
                    .offset(attribute.offset);
                vertex_attributes.push(attribute_info);
            }

            let binding_info = vk::VertexInputBindingDescription::default()
                .binding(i as u32)
                .input_rate(binding.input_rate)
                .stride(binding.stride);
            vertex_bindings.push(binding_info);
        }

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&vertex_attributes)
            .vertex_binding_descriptions(&vertex_bindings);

        let assembly_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vertex_state.topology)
            .primitive_restart_enable(vertex_state.primitive_restart_enable);

        // Fill multisample state
        // TODO
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Fill rasterization state
        let raster = info.rasterization;
        let mut rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .cull_mode(raster.cull_mode)
            .front_face(raster.front_face)
            .rasterizer_discard_enable(raster.discard_enable)
            .depth_bias_enable(false)
            .depth_clamp_enable(false);

        rasterization_state = match raster.polygon_mode {
            PolygonMode::Fill => rasterization_state
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0),
            PolygonMode::Line(width) => rasterization_state
                .polygon_mode(vk::PolygonMode::LINE)
                .line_width(width),
            PolygonMode::Point(width) => rasterization_state
                .polygon_mode(vk::PolygonMode::POINT)
                .line_width(width),
        };

        // Fill stages
        let stages = info
            .stages
            .iter()
            .map(|s| {
                vk::PipelineShaderStageCreateInfo::default()
                    .module(s.module.module)
                    .name(s.entrypoint)
                    .flags(s.flags)
                    .stage(s.stage)
            })
            .collect::<Vec<_>>();

        // Fill viewport

        let viewport_state = match info.viewport {
            PipelineViewportState::Dynamic {
                viewport_count,
                scissor_count,
            } => vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(viewport_count)
                .scissor_count(scissor_count),
            PipelineViewportState::Static {
                viewports,
                scissors,
            } => vk::PipelineViewportStateCreateInfo::default()
                .viewports(viewports)
                .scissors(scissors),
        };

        pipeline_info = pipeline_info
            .color_blend_state(&color_blend_state)
            .depth_stencil_state(&depth_stencil_state)
            .dynamic_state(&dynamic_state)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&assembly_info)
            .multisample_state(&multisample_state)
            .layout(self.layouts[&info.layout_ref])
            .rasterization_state(&rasterization_state)
            .stages(&stages)
            .viewport_state(&viewport_state);

        if let Some(render_pass_state) = info.render_pass {
            pipeline_info = pipeline_info
                .render_pass(render_pass_state.render_pass)
                .subpass(render_pass_state.subpass);
        } else {
            pipeline_info = pipeline_info
                .subpass(0)
                .push_next(&mut rendering_info);
        }

        let pipeline = unsafe {
            match self.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            ) {
                Ok(p) => p[0],
                Err((_, e)) => return Err(VklError::VulkanError(e)),
            }
        };

        let handle = self.append_pipeline(Pipeline::Render(pipeline));

        return Ok(handle);
    }

    /// Frees pipeline resources.
    /// 
    /// **WARNING: Invalidates all references**.
    pub fn clear(&mut self) {
        for (_, pipeline) in self.pipelines.iter() {
            unsafe {
                self.device
                    .destroy_pipeline(
                        pipeline.pipeline(),
                        None,
                    );
            }
        }

        for (_, layout) in self.layouts.iter() {
            unsafe {
                self.device
                    .destroy_pipeline_layout(*layout, None);
            }
        }

        for (_, render_pass) in self.render_passes.iter() {
            unsafe {
                self.device
                    .destroy_render_pass(*render_pass, None);
            }
        }

        self.layouts.clear();
        self.pipelines.clear();
        self.render_passes.clear();

        self.last_layout = 0;
        self.last_pipeline = 0;
        self.last_render_pass = 0;
    }

    fn append_pipeline(&mut self, pipeline: Pipeline) -> PipelineHandle {
        let handle = self.last_pipeline;
        self.last_pipeline += 1;

        self.pipelines.insert(handle, pipeline);

        handle
    }
}

impl Drop for PipelineManager {
    fn drop(&mut self) {
        for (_, &pass) in self.render_passes.iter() {
            unsafe {
                self.device.destroy_render_pass(pass, None);
            }
        }

        for (_, pipe) in self.pipelines.iter() {
            unsafe {
                self.device.destroy_pipeline(pipe.pipeline(), None);
            }
        }

        for (_, &layout) in self.layouts.iter() {
            unsafe {
                self.device.destroy_pipeline_layout(layout, None);
            }
        }
    }
}
