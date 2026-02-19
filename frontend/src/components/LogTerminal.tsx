"use client";

/**
 * LogTerminal - Real-time log viewer component.
 *
 * Displays live engine logs with:
 * - Color-coded log levels
 * - Auto-scroll with pause
 * - Level filtering
 * - Search/highlight
 *
 * US-G10.1: Live Engine Log Monitor
 * Rule #2: Log buffer limited to 500 entries
 */

import React, { useRef, useEffect, useState } from 'react';
import { useWebSocketLogs, LogEntry } from '@/hooks/useWebSocketLogs';
import {
  Terminal, Wifi, WifiOff, Trash2, Pause, Play, Search,
  AlertCircle, Info, AlertTriangle, Bug, Zap
} from 'lucide-react';

interface LogTerminalProps {
  maxHeight?: string;
  showToolbar?: boolean;
}

// Log level colors
const levelColors: Record<string, string> = {
  DEBUG: 'text-gray-500',
  INFO: 'text-emerald-400',
  WARNING: 'text-amber-400',
  ERROR: 'text-red-400',
  CRITICAL: 'text-red-600 font-bold',
};

const levelIcons: Record<string, React.ReactNode> = {
  DEBUG: <Bug size={12} />,
  INFO: <Info size={12} />,
  WARNING: <AlertTriangle size={12} />,
  ERROR: <AlertCircle size={12} />,
  CRITICAL: <Zap size={12} />,
};

export function LogTerminal({ maxHeight = '400px', showToolbar = true }: LogTerminalProps) {
  const [filterLevel, setFilterLevel] = useState<'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'>('INFO');
  const [isPaused, setIsPaused] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  const { logs, isConnected, error, connect, disconnect, clearLogs } = useWebSocketLogs({
    filterLevel,
    autoConnect: true,
  });

  // Auto-scroll when new logs arrive (unless paused)
  useEffect(() => {
    if (!isPaused && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, isPaused]);

  // Filter logs by search term
  const filteredLogs = searchTerm
    ? logs.filter((log) =>
        log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
        log.module.toLowerCase().includes(searchTerm.toLowerCase())
      )
    : logs;

  const highlightText = (text: string, term: string): React.ReactNode => {
    if (!term) return text;
    const parts = text.split(new RegExp(`(${term})`, 'gi'));
    return parts.map((part, i) =>
      part.toLowerCase() === term.toLowerCase() ? (
        <span key={i} className="bg-yellow-500/30 text-yellow-200 px-0.5 rounded">
          {part}
        </span>
      ) : (
        part
      )
    );
  };

  return (
    <div className="flex flex-col bg-[#0d1117] border border-gray-800 rounded-2xl overflow-hidden">
      {/* Toolbar */}
      {showToolbar && (
        <div className="flex items-center justify-between px-4 py-3 bg-gray-900/50 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <Terminal size={16} className="text-forge-crimson" />
            <span className="text-xs font-bold uppercase tracking-widest text-gray-400">
              Engine Logs
            </span>
            <div className="flex items-center gap-1.5">
              {isConnected ? (
                <>
                  <Wifi size={12} className="text-emerald-400" />
                  <span className="text-[10px] text-emerald-400 font-medium">LIVE</span>
                </>
              ) : (
                <>
                  <WifiOff size={12} className="text-red-400" />
                  <span className="text-[10px] text-red-400 font-medium">OFFLINE</span>
                </>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Search */}
            <div className="relative">
              <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-gray-500" />
              <input
                type="text"
                placeholder="Filter..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-32 bg-gray-800 border border-gray-700 rounded-lg pl-7 pr-2 py-1 text-xs focus:border-forge-crimson outline-none"
              />
            </div>

            {/* Level Filter */}
            <select
              value={filterLevel}
              onChange={(e) => setFilterLevel(e.target.value as any)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-2 py-1 text-xs focus:border-forge-crimson outline-none cursor-pointer"
            >
              <option value="DEBUG">DEBUG+</option>
              <option value="INFO">INFO+</option>
              <option value="WARNING">WARN+</option>
              <option value="ERROR">ERROR+</option>
            </select>

            {/* Pause/Play */}
            <button
              onClick={() => setIsPaused(!isPaused)}
              className={`p-1.5 rounded-lg transition-colors ${
                isPaused
                  ? 'bg-amber-500/20 text-amber-400'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
              title={isPaused ? 'Resume auto-scroll' : 'Pause auto-scroll'}
            >
              {isPaused ? <Play size={14} /> : <Pause size={14} />}
            </button>

            {/* Clear */}
            <button
              onClick={clearLogs}
              className="p-1.5 rounded-lg bg-gray-800 text-gray-400 hover:text-white transition-colors"
              title="Clear logs"
            >
              <Trash2 size={14} />
            </button>

            {/* Connect/Disconnect */}
            <button
              onClick={isConnected ? disconnect : connect}
              className={`px-3 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all ${
                isConnected
                  ? 'bg-gray-800 text-gray-400 hover:text-red-400'
                  : 'bg-forge-crimson text-white hover:bg-forge-crimson/80'
              }`}
            >
              {isConnected ? 'Disconnect' : 'Connect'}
            </button>
          </div>
        </div>
      )}

      {/* Log Content */}
      <div
        ref={scrollRef}
        className="font-mono text-xs overflow-auto p-4 space-y-0.5"
        style={{ maxHeight }}
      >
        {error && (
          <div className="flex items-center gap-2 text-red-400 py-2">
            <AlertCircle size={14} />
            <span>{error}</span>
          </div>
        )}

        {filteredLogs.length === 0 ? (
          <div className="text-gray-600 text-center py-8 italic">
            {isConnected ? 'Waiting for logs...' : 'Not connected'}
          </div>
        ) : (
          filteredLogs.map((log, index) => (
            <LogLine key={index} log={log} searchTerm={searchTerm} highlightText={highlightText} />
          ))
        )}
      </div>

      {/* Status Bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-900/30 border-t border-gray-800 text-[10px] text-gray-500">
        <span>{filteredLogs.length} entries</span>
        {isPaused && <span className="text-amber-400">Auto-scroll paused</span>}
        <span>Buffer: {logs.length}/500</span>
      </div>
    </div>
  );
}

// Extracted LogLine component for performance
function LogLine({
  log,
  searchTerm,
  highlightText,
}: {
  log: LogEntry;
  searchTerm: string;
  highlightText: (text: string, term: string) => React.ReactNode;
}) {
  return (
    <div className="flex items-start gap-2 py-0.5 hover:bg-gray-800/30 rounded px-1 -mx-1 group">
      <span className="text-gray-600 shrink-0">{log.timestamp}</span>
      <span className={`shrink-0 flex items-center gap-1 ${levelColors[log.level]}`}>
        {levelIcons[log.level]}
        <span className="w-12">{log.level.padEnd(8)}</span>
      </span>
      <span className="text-gray-500 shrink-0">[{log.module}]</span>
      <span className="text-gray-300 break-all">
        {highlightText(log.message, searchTerm)}
      </span>
    </div>
  );
}

export default LogTerminal;
