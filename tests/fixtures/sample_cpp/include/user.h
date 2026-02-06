#ifndef USER_H
#define USER_H

#include <string>
#include "utils.h"

namespace models {

struct UserData {
    int id;
    std::string name;
    std::string email;
};

class User {
public:
    User(int id, const std::string& name);
    ~User();
    
    int getId() const;
    std::string getName() const;
    void setName(const std::string& name);
    
    static User* create(const std::string& name);

private:
    UserData data_;
    void validate();
};

class AdminUser : public User {
public:
    AdminUser(int id, const std::string& name, int level);
    int getLevel() const;
    
private:
    int admin_level_;
};

} // namespace models

#endif // USER_H
