#include <iostream>
#include "user.h"
#include "utils.h"

#define APP_NAME "UserManager"
#define VERSION "1.0.0"

using namespace models;
using namespace utils;

void print_user(const User& user) {
    std::cout << "User: " << user.getName() << " (ID: " << user.getId() << ")" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << APP_NAME << " v" << VERSION << std::endl;
    
    // Create a user
    User* user = User::create("John Doe");
    print_user(*user);
    
    // Test hash
    int hash = calculate_hash(user->getName());
    std::cout << "Name hash: " << hash << std::endl;
    
    // Test max
    int max = max_value(10, 20);
    std::cout << "Max value: " << max << std::endl;
    
    delete user;
    return 0;
}
