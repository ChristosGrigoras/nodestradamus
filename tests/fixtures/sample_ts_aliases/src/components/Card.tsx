import { cn, formatDate } from "@lib/utils";

interface CardProps {
  title: string;
  date: Date;
  className?: string;
}

export function Card({ title, date, className }: CardProps) {
  return (
    <div className={cn("border rounded p-4", className)}>
      <h2>{title}</h2>
      <span>{formatDate(date)}</span>
    </div>
  );
}
