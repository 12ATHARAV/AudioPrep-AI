import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

export default function FileUpload({ onFileSelected }) {
    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles.length > 0) {
            onFileSelected(acceptedFiles[0]);
        }
    }, [onFileSelected]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'application/pdf': ['.pdf'] },
        maxFiles: 1,
        maxSize: 50 * 1024 * 1024, // 50MB
    });

    return (
        <div className="upload-section">
            <div
                {...getRootProps()}
                className={`dropzone ${isDragActive ? 'active' : ''}`}
            >
                <input {...getInputProps()} />
                <div className="dropzone-content">
                    <span className="upload-icon">📄</span>
                    {isDragActive ? (
                        <p>Drop your PDF here...</p>
                    ) : (
                        <>
                            <p>Drag & drop your PDF notes here</p>
                            <p className="hint">or click to browse</p>
                            <p className="formats">Supports: Docker, OS, CN, OOP, ML, LLM, etc notes</p>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}