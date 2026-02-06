/**
 * Sample user service for testing dependency analysis.
 */

import { helper } from '../utils';

interface User {
    id: string;
    name: string;
}

export class UserService {
    private users: Map<string, User> = new Map();
    
    getUser(id: string): User | undefined {
        helper();
        return this.users.get(id);
    }
    
    createUser(id: string, name: string): User {
        const user: User = { id, name };
        this.users.set(id, user);
        return user;
    }
}

export class AdminService extends UserService {
    deleteUser(id: string): void {
        // Admin can delete users
    }
}
