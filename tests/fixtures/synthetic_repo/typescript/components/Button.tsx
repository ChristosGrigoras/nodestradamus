/**
 * Button component - 1 function.
 *
 * Expected counts:
 * - Functions: 1 (Button)
 * - Classes: 0
 */

interface ButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
}

export function Button({ label, onClick, disabled }: ButtonProps): JSX.Element {
  return (
    <button onClick={onClick} disabled={disabled}>
      {label}
    </button>
  );
}
