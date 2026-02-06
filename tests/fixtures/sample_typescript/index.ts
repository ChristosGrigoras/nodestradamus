/**
 * Sample TypeScript file for testing dependency analysis.
 */

import { UserService } from './services/user';
import { helper } from './utils';

export function main(): void {
    const service = new UserService();
    service.getUser('1');
    helper();
}

export const VERSION = '1.0.0';
