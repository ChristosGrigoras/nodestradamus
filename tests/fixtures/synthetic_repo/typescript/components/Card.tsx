/**
 * Card component - 2 functions.
 *
 * Expected counts:
 * - Functions: 2 (Card, CardHeader)
 * - Classes: 0
 */

interface CardProps {
  title: string;
  children: React.ReactNode;
}

export function Card({ title, children }: CardProps): JSX.Element {
  return (
    <div className="card">
      <CardHeader title={title} />
      <div className="card-body">{children}</div>
    </div>
  );
}

function CardHeader({ title }: { title: string }): JSX.Element {
  return <h2 className="card-header">{title}</h2>;
}
