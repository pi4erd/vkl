use std::fmt::Display;

use ash::vk;
use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DebugUtilsMessageSeverity: u32 {
        const VERBOSE = vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE.as_raw();
        const INFO = vk::DebugUtilsMessageSeverityFlagsEXT::INFO.as_raw();
        const WARNING = vk::DebugUtilsMessageSeverityFlagsEXT::WARNING.as_raw();
        const ERROR = vk::DebugUtilsMessageSeverityFlagsEXT::ERROR.as_raw();
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DebugUtilsMessageType: u32 {
        const GENERAL = vk::DebugUtilsMessageTypeFlagsEXT::GENERAL.as_raw();
        const VALIDATION = vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION.as_raw();
        const PERFORMANCE = vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE.as_raw();
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

impl Display for DebugUtilsMessageSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bit_count = self.iter().count();
        for (i, (name, _)) in self.iter_names().enumerate() {
            write!(f, "{}", name)?;

            if i < bit_count - 1 {
                write!(f, " | ")?;
            }
        }

        Ok(())
    }
}

impl Display for DebugUtilsMessageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bit_count = self.iter().count();
        for (i, (name, _)) in self.iter_names().enumerate() {
            write!(f, "{}", name)?;

            if i < bit_count - 1 {
                write!(f, " | ")?;
            }
        }

        Ok(())
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

#[cfg(test)]
mod tests {
    use crate::debug_utils::*;

    #[test]
    fn severity_type_display_test() {
        let single_sev = DebugUtilsMessageSeverity::ERROR;
        assert_eq!(format!("{}", single_sev), "ERROR");

        let multi_sev = DebugUtilsMessageSeverity::WARNING |
            DebugUtilsMessageSeverity::ERROR;
        // NOTE: Order is the same as defined in struct
        assert_eq!(format!("{}", multi_sev), "WARNING | ERROR");

        let single_type = DebugUtilsMessageType::GENERAL;
        assert_eq!(format!("{}", single_type), "GENERAL");

        let multi_type = DebugUtilsMessageType::GENERAL |
            DebugUtilsMessageType::VALIDATION;
        assert_eq!(format!("{}", multi_type), "GENERAL | VALIDATION");
    }
}
