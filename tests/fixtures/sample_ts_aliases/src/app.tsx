import { Button } from "@components/Button";
import { Card } from "@/components/Card";
import { cn } from "@/lib/utils";

export function App() {
  return (
    <div className={cn("container mx-auto")}>
      <Card title="Welcome" date={new Date()} />
      <Button onClick={() => console.log("clicked")}>Click me</Button>
    </div>
  );
}
