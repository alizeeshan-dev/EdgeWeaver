export function LoadingState({ label = "Loading research data" }: { label?: string }) {
  return (
    <div className="state-card" role="status">
      <span className="loading-ring" />
      <span>{label}…</span>
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state-card error" role="alert">
      <strong>Unable to load data</strong>
      <span>{message}</span>
      {onRetry ? (
        <button className="outline-button" type="button" onClick={onRetry}>
          Try again
        </button>
      ) : null}
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return <div className="state-card">{message}</div>;
}
