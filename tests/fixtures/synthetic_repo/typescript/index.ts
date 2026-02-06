/**
 * Entry point - 2 functions.
 *
 * Expected counts:
 * - Functions: 2 (main, bootstrap)
 * - Classes: 0
 */

import { initApi } from "./lib/api";
import { formatDate } from "./lib/utils";

export function main(): void {
  console.log("Starting application...");
  bootstrap();
}

async function bootstrap(): Promise<void> {
  await initApi();
  console.log(`Started at ${formatDate(new Date())}`);
}
