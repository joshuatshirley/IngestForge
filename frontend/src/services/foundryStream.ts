/**
 * Foundry Streaming Client (US-3102.1)
 *
 * TypeScript client for consuming Server-Sent Events (SSE) from the
 * Streaming Foundry API.
 *
 * @module foundryStream
 */

// -------------------------------------------------------------------------
// Type Definitions
// -------------------------------------------------------------------------

export interface Citation {
  source: string;
  page?: number;
}

export interface ProgressMetadata {
  current: number;
  total: number;
  stage: string;
  percentage: number;
  message?: string;
}

export interface ChunkMetadata {
  start_char: number;
  end_char: number;
}

export interface ChunkEvent {
  chunk_id: string;
  content: string;
  citations: Citation[];
  progress: ProgressMetadata;
  is_final: boolean;
  metadata?: ChunkMetadata;
}

export interface ProgressEvent {
  current: number;
  total: number;
  stage: string;
  percentage: number;
  message: string;
}

export interface ErrorEvent {
  error_code: string;
  message: string;
  suggestion: string;
  details: Record<string, unknown>;
}

export interface CompleteEvent {
  total_chunks: number;
  success: boolean;
  summary: string;
}

export type StreamEventType =
  | { type: 'chunk'; data: ChunkEvent }
  | { type: 'progress'; data: ProgressEvent }
  | { type: 'error'; data: ErrorEvent }
  | { type: 'complete'; data: CompleteEvent }
  | { type: 'keepalive'; data: { message: string } };

// -------------------------------------------------------------------------
// Streaming Client
// -------------------------------------------------------------------------

export interface FoundryConfig {
  [key: string]: unknown;
}

/**
 * Stream foundry execution via SSE.
 *
 * US-3102.1: Core streaming client.
 *
 * @param filePath - Path to document to process
 * @param config - Optional pipeline configuration
 * @yields Stream events as they arrive
 *
 * @example
 * ```typescript
 * const stream = streamFoundry('document.pdf');
 * for await (const event of stream) {
 *   if (event.type === 'chunk') {
 *     console.log('Chunk:', event.data.chunk_id);
 *   }
 * }
 * ```
 */
export async function* streamFoundry(
  filePath: string,
  config?: FoundryConfig
): AsyncGenerator<StreamEventType, void, void> {
  const response = await fetch('/v1/foundry/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file_path: filePath,
      config: config || null,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  if (!response.body) {
    throw new Error('Response body is null');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  // JPL Rule #2: Bounded loops - prevent infinite streaming
  const MAX_STREAM_EVENTS = 100000; // ~6.4GB at 64KB avg per event
  const STREAM_TIMEOUT_MS = 300000; // 5 minutes max stream duration

  const startTime = Date.now();
  let iterations = 0;

  try {
    while (iterations < MAX_STREAM_EVENTS) {
      // JPL Rule #2: Check timeout bound
      if (Date.now() - startTime > STREAM_TIMEOUT_MS) {
        throw new Error('Stream timeout exceeded (5 minutes)');
      }

      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE events (terminated by double newline)
      const events = buffer.split('\n\n');
      buffer = events.pop() || ''; // Keep last incomplete event

      for (const eventText of events) {
        if (!eventText.trim()) {
          continue;
        }

        const parsedEvent = parseSSEEvent(eventText);
        if (parsedEvent) {
          yield parsedEvent;
        }
      }

      iterations++;
    }

    // JPL Rule #2: Assert loop terminated within bounds
    if (iterations >= MAX_STREAM_EVENTS) {
      throw new Error(`Stream exceeded maximum event limit (${MAX_STREAM_EVENTS})`);
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Parse SSE event text.
 *
 * @param eventText - Raw SSE event text
 * @returns Parsed StreamEventType or null
 */
function parseSSEEvent(eventText: string): StreamEventType | null {
  const lines = eventText.split('\n');
  let eventType = 'chunk'; // Default type
  let dataLine: string | null = null;

  for (const line of lines) {
    if (line.startsWith('event: ')) {
      eventType = line.slice(7).trim();
    } else if (line.startsWith('data: ')) {
      dataLine = line.slice(6).trim();
    }
  }

  if (!dataLine) {
    return null;
  }

  try {
    const data = JSON.parse(dataLine);

    switch (eventType) {
      case 'chunk':
        return { type: 'chunk', data: data as ChunkEvent };
      case 'progress':
        return { type: 'progress', data: data as ProgressEvent };
      case 'error':
        return { type: 'error', data: data as ErrorEvent };
      case 'complete':
        return { type: 'complete', data: data as CompleteEvent };
      case 'keepalive':
        return { type: 'keepalive', data: data as { message: string } };
      default:
        console.warn(`Unknown event type: ${eventType}`);
        return null;
    }
  } catch (e) {
    console.error('Failed to parse SSE event:', e);
    return null;
  }
}

/**
 * Collect all chunks from stream.
 *
 * Utility function to consume entire stream and return all chunks.
 *
 * @param filePath - Path to document
 * @param config - Optional config
 * @returns Array of all chunk events
 *
 * @example
 * ```typescript
 * const chunks = await streamAllChunks('document.pdf');
 * console.log(`Received ${chunks.length} chunks`);
 * ```
 */
export async function streamAllChunks(
  filePath: string,
  config?: FoundryConfig
): Promise<ChunkEvent[]> {
  const chunks: ChunkEvent[] = [];

  for await (const event of streamFoundry(filePath, config)) {
    if (event.type === 'chunk') {
      chunks.push(event.data);
    } else if (event.type === 'error') {
      throw new Error(`Stream error: ${event.data.message}`);
    }
  }

  return chunks;
}

/**
 * Stream with progress callback.
 *
 * @param filePath - Path to document
 * @param onProgress - Progress callback
 * @param config - Optional config
 * @returns Array of chunk events
 */
export async function streamWithProgress(
  filePath: string,
  onProgress: (progress: ProgressMetadata) => void,
  config?: FoundryConfig
): Promise<ChunkEvent[]> {
  const chunks: ChunkEvent[] = [];

  for await (const event of streamFoundry(filePath, config)) {
    if (event.type === 'chunk') {
      chunks.push(event.data);
      onProgress(event.data.progress);
    } else if (event.type === 'progress') {
      onProgress(event.data);
    } else if (event.type === 'error') {
      throw new Error(`Stream error: ${event.data.message}`);
    }
  }

  return chunks;
}
