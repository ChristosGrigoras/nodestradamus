/**
 * Sample utility functions for testing dependency analysis.
 */

export function helper(): void {
    console.log('Helper called');
}

export const formatName = (name: string): string => {
    return name.toUpperCase();
};

export async function fetchData(url: string): Promise<unknown> {
    const response = await fetch(url);
    return response.json();
}
