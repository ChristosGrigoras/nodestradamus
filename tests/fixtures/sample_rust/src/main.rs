// Main entry point for the sample Rust application

mod utils;
mod services;

use crate::utils::helpers::format_message;
use crate::services::user::UserService;

fn main() {
    let service = UserService::new();
    let message = format_message("Hello, World!");
    println!("{}", message);
    service.greet();
}

fn helper_function(name: &str) -> String {
    format!("Helper says: {}", name)
}

pub struct Config {
    pub debug: bool,
    pub version: String,
}

impl Config {
    pub fn new() -> Self {
        Config {
            debug: false,
            version: String::from("1.0.0"),
        }
    }

    pub fn is_debug(&self) -> bool {
        self.debug
    }
}
