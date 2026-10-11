import { useState } from 'react';
import FileUpload from './components/FileUpload';
import PodcastConfig from './components/PodcastConfig';
import StatusTracker from './components/StatusTracker';
import AudioPlayer from './components/AudioPlayer';
import './App.css';

function App() {
  const [step, setStep] = useState('upload'); // upload, config, processing, done
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);

  return (
    <div className="app">
      <header className="header">
        <h1>🎙️ AudioPrep AI</h1>
        <p>Turn Technical Notes into Interactive AI Podcasts</p>
      </header>

      <main className="main">
        {step === 'upload' && (
          <FileUpload
            onFileSelected={(f) => {
              setFile(f);
              setStep('config');
            }}
          />
        )}

        {step === 'config' && (
          <PodcastConfig
            filename={file?.name}
            onGenerate={(config) => {
              handleGenerate(file, config, setJobId, setStep);
            }}
            onBack={() => setStep('upload')}
          />
        )}

        {step === 'processing' && (
          <StatusTracker
            jobId={jobId}
            onComplete={(url) => {
              setAudioUrl(url);
              setStep('done');
            }}
          />
        )}

        {step === 'done' && (
          <AudioPlayer
            audioUrl={audioUrl}
            onNewPodcast={() => {
              setStep('upload');
              setFile(null);
              setJobId(null);
              setAudioUrl(null);
            }}
          />
        )}
      </main>
    </div>
  );
}

async function handleGenerate(file, config, setJobId, setStep) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('difficulty', config.difficulty);
  formData.append('podcast_length', config.podcastLength);
  formData.append('focus', config.focus);
  formData.append('chunking_strategy', config.chunkingStrategy);

  const queryParams = new URLSearchParams({
    difficulty: config.difficulty,
    podcast_length: config.podcastLength,
    focus: config.focus,
    chunking_strategy: config.chunkingStrategy,
  });

  try {
    const response = await fetch(`/api/podcasts?${queryParams.toString()}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned ${response.status}`);
    }

    const data = await response.json();
    setJobId(data.job_id);
    setStep('processing');
  } catch (error) {
    console.error('Upload failed:', error);
    alert(`Failed to start podcast generation: ${error.message}`);
  }
}

export default App;