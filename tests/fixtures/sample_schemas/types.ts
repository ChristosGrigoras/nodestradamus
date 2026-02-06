// Sample TypeScript types for testing field extraction

interface User {
    id: number;
    email: string;
    name?: string;
    isActive: boolean;
}

type Order = {
    id: number;
    userId: number;
    total: number;
    status: string;
};

export { User, Order };
