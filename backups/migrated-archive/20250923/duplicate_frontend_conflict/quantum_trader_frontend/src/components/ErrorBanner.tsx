<<<<<<< Updated upstream
// frontend/src/components/ErrorBanner.tsx
type Props = { show?: boolean; message?: string };

export default function ErrorBanner({ show, message }: Props) {
=======
<<<<<<<< Updated upstream:frontend/src/components/ErrorBanner.jsx
// Auto-generated re-export stub
export { default } from './ErrorBanner.tsx';
========
// frontend/src/components/ErrorBanner.tsx
export default function ErrorBanner({ show, message }) {
>>>>>>> Stashed changes
  if (!show) return null;

  return (
    <div className="bg-red-600 text-white text-center py-2">
      ⚠️ {message}
    </div>
  );
}
<<<<<<< Updated upstream
=======
>>>>>>>> Stashed changes:frontend/src/components/ErrorBanner.tsx
>>>>>>> Stashed changes
