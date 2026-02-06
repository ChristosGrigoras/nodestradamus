// Helper utilities

use std::fmt;

pub fn format_message(msg: &str) -> String {
    format!("[INFO] {}", msg)
}

pub fn uppercase(s: &str) -> String {
    s.to_uppercase()
}

pub struct Formatter {
    prefix: String,
}

impl Formatter {
    pub fn new(prefix: &str) -> Self {
        Formatter {
            prefix: prefix.to_string(),
        }
    }

    pub fn format(&self, msg: &str) -> String {
        format!("{}: {}", self.prefix, msg)
    }
}

impl fmt::Display for Formatter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Formatter({})", self.prefix)
    }
}
