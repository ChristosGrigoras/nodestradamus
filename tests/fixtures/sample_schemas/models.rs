//! Sample Rust structs for testing field extraction

pub struct User {
    pub id: i32,
    pub email: String,
    pub name: Option<String>,
    pub is_active: bool,
}

pub struct Order {
    pub id: i32,
    pub user_id: i32,
    pub total: f64,
    pub status: String,
}
