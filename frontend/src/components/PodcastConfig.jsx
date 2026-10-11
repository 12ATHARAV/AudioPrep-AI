import { useState } from 'react';

export default function PodcastConfig({ filename, onGenerate, onBack }) {
    const [config, setConfig] = useState({
        difficulty: 'intermediate',
        podcastLength: 120,
        focus: 'general',
        chunkingStrategy: 'structure_aware',
    });

    return (
        <div className="config-section">
            <h2>Configure Podcast</h2>
            <p className="filename">📄 {filename}</p>

            <div className="config-form">
                <label>
                    Difficulty Level
                    <select
                        value={config.difficulty}
                        onChange={(e) => setConfig({ ...config, difficulty: e.target.value })}
                    >
                        <option value="beginner">🟢 Beginner</option>
                        <option value="intermediate">🟡 Intermediate</option>
                        <option value="advanced">🔴 Advanced</option>
                    </select>
                </label>

                <label>
                    Podcast Length
                    <select
                        value={config.podcastLength}
                        onChange={(e) => setConfig({ ...config, podcastLength: parseInt(e.target.value) })}
                    >
                        <option value={120}>2 Hours (120 min - Comprehensive Full Podcast)</option>
                        <option value={60}>1 Hour (60 min - In-Depth Episode)</option>
                        <option value={30}>30 minutes (Standard Episode)</option>
                        <option value={20}>20 minutes (Executive Summary)</option>
                        <option value={10}>10 minutes (Quick Summary - Key Topics)</option>
                    </select>
                </label>

                <label>
                    Focus
                    <select
                        value={config.focus}
                        onChange={(e) => setConfig({ ...config, focus: e.target.value })}
                    >
                        <option value="general">📚 General Learning</option>
                        <option value="interview">💼 Interview Prep</option>
                        <option value="deep-dive">🔬 Deep Dive</option>
                    </select>
                </label>

                <label>
                    RAG Strategy
                    <select
                        value={config.chunkingStrategy}
                        onChange={(e) => setConfig({ ...config, chunkingStrategy: e.target.value })}
                    >
                        <option value="structure_aware">Structure-Aware (Recommended)</option>
                        <option value="semantic">Semantic</option>
                        <option value="parent_child">Parent-Child</option>
                        <option value="recursive">Recursive (Baseline)</option>
                    </select>
                </label>
            </div>

            <div className="config-actions">
                <button className="btn-secondary" onClick={onBack}>← Back</button>
                <button className="btn-primary" onClick={() => onGenerate(config)}>
                    🎙️ Generate Podcast
                </button>
            </div>
        </div>
    );
}