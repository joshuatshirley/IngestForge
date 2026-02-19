/**
 * Streaming Progress Component (US-3102.1 / US-1104.1)
 *
 * Real-time progress display for streaming foundry API.
 * Shows progress bar, stage indicator, and chunk previews.
 * Enhanced with Pause/Resume controls.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { streamFoundry, ChunkEvent, ProgressMetadata, StreamEventType } from '../../services/foundryStream';
import { usePauseFoundryStreamMutation, useResumeFoundryStreamMutation } from '@/store/api/ingestforgeApi';
import { PauseIcon, PlayIcon, StopIcon } from '@heroicons/react/outline';

// -------------------------------------------------------------------------
// Type Definitions
// -------------------------------------------------------------------------

export interface StreamingProgressProps {
  filePath: string;
  onComplete?: (chunks: ChunkEvent[]) => void;
  onError?: (error: Error) => void;
  onCancel?: () => void;
}

export interface StreamState {
  isStreaming: boolean;
  progress: ProgressMetadata | null;
  chunks: ChunkEvent[];
  error: string | null;
}

// -------------------------------------------------------------------------
// Component
// -------------------------------------------------------------------------

export const StreamingProgress: React.FC<StreamingProgressProps> = ({
  filePath,
  onComplete,
  onError,
  onCancel,
}) => {
  const [state, setState] = useState<StreamState>({
    isStreaming: false,
    progress: null,
    chunks: [],
    error: null,
  });

  const [streamId, setStreamId] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  const [pauseStream] = usePauseFoundryStreamMutation();
  const [resumeStream] = useResumeFoundryStreamMutation();

  // Start streaming on mount
  useEffect(() => {
    startStreaming();
    return () => {
      // Cleanup on unmount
      if (abortController) {
        abortController.abort();
      }
    };
  }, [filePath]);

  // JPL Rule #4: Extract event handlers to keep functions ≤60 lines
  const handleStreamEvent = useCallback(
    (
      event: StreamEventType,
      chunks: ChunkEvent[],
      controller: AbortController
    ): boolean => {
      if (controller.signal.aborted) {
        return false; // Stop processing
      }

      switch (event.type) {
        case 'chunk':
          chunks.push(event.data);
          setState(prev => ({
            ...prev,
            chunks: [...prev.chunks, event.data],
            progress: event.data.progress,
          }));
          break;

        case 'progress':
          // US-1104.1: Capture stream_id for controls
          if ((event.data as any).stream_id) {
            setStreamId((event.data as any).stream_id);
          }
          setState(prev => ({
            ...prev,
            progress: event.data,
          }));
          break;

        case 'error':
          setState(prev => ({
            ...prev,
            error: event.data.message,
            isStreaming: false,
          }));
          if (onError) {
            onError(new Error(event.data.message));
          }
          return false; // Stop processing

        case 'complete':
          setState(prev => ({
            ...prev,
            isStreaming: false,
          }));
          if (onComplete) {
            onComplete(chunks);
          }
          return false; // Stop processing

        case 'keepalive':
          // Ignore keepalive events
          break;
      }

      return true; // Continue processing
    },
    [onComplete, onError]
  );

  const startStreaming = useCallback(async () => {
    const controller = new AbortController();
    setAbortController(controller);

    setState(prev => ({
      ...prev,
      isStreaming: true,
      error: null,
    }));

    try {
      const chunks: ChunkEvent[] = [];

      for await (const event of streamFoundry(filePath)) {
        const shouldContinue = handleStreamEvent(event, chunks, controller);
        if (!shouldContinue) {
          break;
        }
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setState(prev => ({
        ...prev,
        error: errorMessage,
        isStreaming: false,
      }));
      if (onError) {
        onError(error instanceof Error ? error : new Error(errorMessage));
      }
    }
  }, [filePath, handleStreamEvent, onError]);

  const handleCancel = useCallback(() => {
    if (abortController) {
      abortController.abort();
    }
    setState(prev => ({
      ...prev,
      isStreaming: false,
    }));
    if (onCancel) {
      onCancel();
    }
  }, [abortController, onCancel]);

  const handlePause = async () => {
    if (streamId) {
      try {
        await pauseStream(streamId).unwrap();
        setIsPaused(true);
      } catch (err) {
        console.error('Failed to pause stream:', err);
      }
    }
  };

  const handleResume = async () => {
    if (streamId) {
      try {
        await resumeStream(streamId).unwrap();
        setIsPaused(false);
      } catch (err) {
        console.error('Failed to resume stream:', err);
      }
    }
  };

  // Render progress bar
  const renderProgressBar = () => {
    if (!state.progress) {
      return null;
    }

    const { current, total, percentage } = state.progress;

    return (
      <div className={`mb-4 transition-opacity duration-300 ${isPaused ? 'opacity-50' : 'opacity-100'}`}>
        <div className="flex justify-between text-sm text-gray-600 mb-1">
          <span>
            {current} / {total} chunks
          </span>
          <span>{percentage.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div
            className={`h-2.5 rounded-full transition-all duration-300 ${isPaused ? 'bg-yellow-500' : 'bg-blue-600'}`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    );
  };

  // Render stage indicator
  const renderStageIndicator = () => {
    if (!state.progress) {
      return null;
    }

    const { stage, message } = state.progress;

    return (
      <div className="mb-4 flex items-center text-sm">
        <div className={`inline-block w-2 h-2 rounded-full mr-2 ${isPaused ? 'bg-yellow-500' : 'bg-blue-600 animate-pulse'}`} />
        <span className="font-medium text-gray-700">
          {isPaused ? 'PAUSED' : stage.charAt(0).toUpperCase() + stage.slice(1)}
        </span>
        {message && (
          <span className="ml-2 text-gray-500">
            — {message}
          </span>
        )}
      </div>
    );
  };

  // Render chunk list
  const renderChunkList = () => {
    if (state.chunks.length === 0) {
      return null;
    }

    return (
      <div className="mt-4 max-h-64 overflow-y-auto border border-gray-300 rounded p-2">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">
          Processed Chunks ({state.chunks.length})
        </h3>
        <ul className="space-y-2">
          {state.chunks.map((chunk, index) => (
            <li
              key={chunk.chunk_id}
              className="text-xs bg-gray-50 p-2 rounded border border-gray-200"
            >
              <div className="font-mono text-gray-600 mb-1">
                {chunk.chunk_id}
              </div>
              <div className="text-gray-800 truncate">
                {chunk.content.substring(0, 100)}
                {chunk.content.length > 100 && '...'}
              </div>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  // Render error message
  const renderError = () => {
    if (!state.error) {
      return null;
    }

    return (
      <div className="mb-4 p-3 bg-red-50 border border-red-300 rounded text-red-800 text-sm">
        <strong>Error:</strong> {state.error}
      </div>
    );
  };

  return (
    <div className="streaming-progress p-4 bg-white rounded shadow">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-bold text-gray-800">
          Processing: {filePath.split('/').pop()}
        </h2>
        <div className="flex gap-2">
          {state.isStreaming && (
            <>
              {isPaused ? (
                <button
                  onClick={handleResume}
                  className="p-2 bg-green-500 text-white rounded hover:bg-green-600 transition-colors"
                  title="Resume"
                >
                  <PlayIcon className="w-5 h-5" />
                </button>
              ) : (
                <button
                  onClick={handlePause}
                  className="p-2 bg-yellow-500 text-white rounded hover:bg-yellow-600 transition-colors"
                  title="Pause"
                >
                  <PauseIcon className="w-5 h-5" />
                </button>
              )}
              <button
                onClick={handleCancel}
                className="p-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
                title="Cancel"
              >
                <StopIcon className="w-5 h-5" />
              </button>
            </>
          )}
        </div>
      </div>

      {renderError()}
      {renderProgressBar()}
      {renderStageIndicator()}
      {renderChunkList()}

      {!state.isStreaming && !state.error && state.chunks.length > 0 && (
        <div className="mt-4 p-3 bg-green-50 border border-green-300 rounded text-green-800 text-sm">
          <strong>Complete!</strong> Processed {state.chunks.length} chunks successfully.
        </div>
      )}
    </div>
  );
};

export default StreamingProgress;
