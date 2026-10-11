import { useState, useEffect } from 'react';

const STAGE_LABELS = {
  QUEUED: '⏳ Queued',
  INGESTING: '📄 Reading PDF...',
  CHUNKING: '✂️ Analyzing Structure...',
  RETRIEVING: '🔍 Finding Key Concepts...',
  PLANNING: '📋 Planning Podcast...',
  GENERATING_DIALOGUE: '💬 Writing Conversation...',
  CRITIQUING: '🔎 Quality Check...',
  REVISING: '✏️ Improving Dialogue...',
  GENERATING_AUDIO: '🔊 Generating Voice...',
  PROCESSING_AUDIO: '🎧 Polishing Audio...',
  COMPLETED: '✅ Done!',
  FAILED: '❌ Failed',
};

export default function StatusTracker({ jobId, onComplete }) {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    if (!jobId) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/podcasts/${jobId}`);
        const data = await res.json();
        setStatus(data);

        if (data.status === 'COMPLETED') {
          clearInterval(interval);
          onComplete(data.audio_url);
        }
        if (data.status === 'FAILED') {
          clearInterval(interval);
        }
      } catch (error) {
        console.error('Status check failed:', error);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, [jobId, onComplete]);

  return (
    <div className="status-section">
      <h2>Generating Your Podcast</h2>

      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${status?.progress || 10}%` }}
        />
      </div>

      <p className="status-text">
        {STAGE_LABELS[status?.status] || status?.current_stage || '⏳ Starting generation pipeline...'}
      </p>

      {status?.error && (
        <p className="error-text">Error: {status.error}</p>
      )}

      <p className="status-hint">
        This typically takes 1-3 minutes depending on document length and podcast duration.
      </p>
    </div>
  );
}
