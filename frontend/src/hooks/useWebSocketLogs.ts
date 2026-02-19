/**
 * WebSocket hook for real-time log streaming.
 *
 * Connects to the IngestForge API log WebSocket endpoint
 * and provides log entries with automatic reconnection.
 *
 * US-G10.1: Live Engine Log Monitor
 */

import { useState, useEffect, useCallback, useRef } from 'react';

export interface LogEntry {
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
  message: string;
  module: string;
}

interface UseWebSocketLogsOptions {
  maxEntries?: number;
  autoConnect?: boolean;
  filterLevel?: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
}

// Rule #2: Fixed upper bound for log buffer
const DEFAULT_MAX_ENTRIES = 500;

export function useWebSocketLogs(options: UseWebSocketLogsOptions = {}) {
  const {
    maxEntries = DEFAULT_MAX_ENTRIES,
    autoConnect = true,
    filterLevel = 'INFO'
  } = options;

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Level priority for filtering
  const levelPriority: Record<string, number> = {
    DEBUG: 0,
    INFO: 1,
    WARNING: 2,
    ERROR: 3,
    CRITICAL: 4,
  };

  const minLevel = levelPriority[filterLevel] || 1;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/v1/ws/logs';

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const entry: LogEntry = JSON.parse(event.data);

          // Filter by level
          if (levelPriority[entry.level] >= minLevel) {
            setLogs((prev) => {
              const updated = [...prev, entry];
              // Rule #2: Enforce max entries
              if (updated.length > maxEntries) {
                return updated.slice(-maxEntries);
              }
              return updated;
            });
          }
        } catch (e) {
          console.error('Failed to parse log entry:', e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        wsRef.current = null;

        // Auto-reconnect after 3 seconds
        if (autoConnect) {
          reconnectTimeoutRef.current = setTimeout(connect, 3000);
        }
      };

      ws.onerror = () => {
        setError('WebSocket connection failed');
        setIsConnected(false);
      };
    } catch (e) {
      setError('Failed to create WebSocket connection');
    }
  }, [autoConnect, maxEntries, minLevel]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    logs,
    isConnected,
    error,
    connect,
    disconnect,
    clearLogs,
  };
}
