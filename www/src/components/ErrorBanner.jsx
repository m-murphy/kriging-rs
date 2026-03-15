export default function ErrorBanner({ message, onDismiss }) {
  return (
    <div className="error-banner" role="alert">
      <strong>Error:</strong> {message}
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          style={{ marginLeft: "0.75rem" }}
          aria-label="Dismiss error"
        >
          Dismiss
        </button>
      )}
    </div>
  );
}
