use ash::vk;
use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DebugUtilsMessageSeverity: u32 {
        const VERBOSE = 0b1;
        const INFO = 0b1_0000;
        const WARNING = 0b1_0000_0000;
        const ERROR = 0b1_0000_0000_0000;
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DebugUtilsMessageType: u32 {
        const GENERAL = 0b1;
        const VALIDATION = 0b10;
        const PERFORMANCE = 0b100;
    }
}

impl Into<vk::DebugUtilsMessageSeverityFlagsEXT> for DebugUtilsMessageSeverity {
    fn into(self) -> vk::DebugUtilsMessageSeverityFlagsEXT {
        vk::DebugUtilsMessageSeverityFlagsEXT::from_raw(self.0.0)
    }
}

impl Into<vk::DebugUtilsMessageTypeFlagsEXT> for DebugUtilsMessageType {
    fn into(self) -> vk::DebugUtilsMessageTypeFlagsEXT {
        vk::DebugUtilsMessageTypeFlagsEXT::from_raw(self.0.0)
    }
}

pub struct DebugUtilsCallbackData<'a> {
    pub message: &'a str,
    // TODO: Expand the class, if needed
}

pub type DebugCallback = fn(
            DebugUtilsMessageSeverity,
            DebugUtilsMessageType,
            &DebugUtilsCallbackData
        ) -> bool;
