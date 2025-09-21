// frontend/src/components/ErrorBanner.tsx
type Props = { show?: boolean; message?: string };

export default function ErrorBanner({ show, message }: Props) {
  if (!show) return null;

  return (
    <div className="bg-red-600 text-white text-center py-2">
      ⚠️ {message}
    </div>
  );
}
