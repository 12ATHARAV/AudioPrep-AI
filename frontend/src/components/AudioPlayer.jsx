export default function AudioPlayer({ audioUrl, onNewPodcast }) {
    const fullUrl = `http://localhost:8000${audioUrl}`;

    return (
        <div className="player-section">
            <h2>🎧 Your Podcast is Ready!</h2>

            <audio controls src={fullUrl} className="audio-player">
                Your browser does not support the audio element.
            </audio>

            <div className="player-actions">
                <a
                    href={fullUrl}
                    download
                    className="btn-primary"
                >
                    ⬇️ Download MP3
                </a>

                <button className="btn-secondary" onClick={onNewPodcast}>
                    📄 Create Another Podcast
                </button>
            </div>
        </div>
    );
}