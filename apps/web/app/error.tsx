"use client";

export default function ErrorPage({ reset }: { error: Error; reset: () => void }) {
  return <main className="boot"><h1>Unable to display the command center</h1><p>Your server-side replay is preserved. Reload this view to reconnect.</p><button className="primary" onClick={reset}>Try again</button></main>;
}
