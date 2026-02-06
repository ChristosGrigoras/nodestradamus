import { cn } from "@/lib/utils";

interface ButtonProps {
  children: React.ReactNode;
  variant?: "primary" | "secondary";
  onClick?: () => void;
}

export function Button({ children, variant = "primary", onClick }: ButtonProps) {
  return (
    <button
      className={cn(
        "px-4 py-2 rounded",
        variant === "primary" ? "bg-blue-500" : "bg-gray-500"
      )}
      onClick={onClick}
    >
      {children}
    </button>
  );
}
