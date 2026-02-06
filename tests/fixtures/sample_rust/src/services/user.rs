// User service

use super::super::utils::helpers::Formatter;

pub struct UserService {
    name: String,
    formatter: Formatter,
}

impl UserService {
    pub fn new() -> Self {
        UserService {
            name: String::from("Default User"),
            formatter: Formatter::new("User"),
        }
    }

    pub fn with_name(name: &str) -> Self {
        UserService {
            name: name.to_string(),
            formatter: Formatter::new("User"),
        }
    }

    pub fn greet(&self) {
        println!("{}", self.formatter.format(&format!("Hello, {}!", self.name)));
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }
}

pub trait Authenticatable {
    fn authenticate(&self) -> bool;
    fn get_token(&self) -> Option<String>;
}

impl Authenticatable for UserService {
    fn authenticate(&self) -> bool {
        true
    }

    fn get_token(&self) -> Option<String> {
        Some(format!("token_{}", self.name))
    }
}

pub enum UserRole {
    Admin,
    User,
    Guest,
}

impl UserRole {
    pub fn is_admin(&self) -> bool {
        matches!(self, UserRole::Admin)
    }
}
