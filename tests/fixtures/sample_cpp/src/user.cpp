#include "user.h"
#include <stdexcept>

namespace models {

static int next_id = 1;

User::User(int id, const std::string& name) {
    data_.id = id;
    data_.name = name;
    validate();
}

User::~User() {
    // Cleanup
}

int User::getId() const {
    return data_.id;
}

std::string User::getName() const {
    return data_.name;
}

void User::setName(const std::string& name) {
    data_.name = name;
    validate();
}

User* User::create(const std::string& name) {
    return new User(next_id++, name);
}

void User::validate() {
    if (data_.name.empty()) {
        throw std::invalid_argument("Name cannot be empty");
    }
}

// AdminUser implementation
AdminUser::AdminUser(int id, const std::string& name, int level)
    : User(id, name), admin_level_(level) {
}

int AdminUser::getLevel() const {
    return admin_level_;
}

} // namespace models
