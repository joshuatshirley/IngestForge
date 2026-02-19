"use client";

import React, { useEffect, useState, useRef } from 'react';
import { Terminal as TerminalIcon, Trash2, Pause, Play, Download } from 'lucide-react';

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  module: string;
}

export const LogTerminal = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    // connect to WebSocket
    ws.current = new WebSocket('ws://localhost:8000/v1/ws/logs');
    
    ws.current.onmessage = (event) => {
      if (isPaused) return;
      const newLog = JSON.parse(event.data);
      setLogs((prev) => [...prev.slice(-499), newLog]);
    };

    return () => {
      ws.current?.close();
    };
  }, [isPaused]);

  useEffect(() => {
    if (!isPaused && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, isPaused]);

  const getLevelStyle = (level: string) => {
    switch (level) {
      case 'ERROR': return 'text-red-400 font-bold';
      case 'WARNING': return 'text-yellow-400';
      case 'DEBUG': return 'text-blue-400 opacity-70';
      default: return 'text-green-400';
    }
  };

  return (
    <div className="forge-card p-0 border-gray-800 bg-black flex flex-col h-full min-h-[400px] overflow-hidden font-mono shadow-2xl">
      <div className="bg-gray-900 px-6 py-3 border-b border-gray-800 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <TerminalIcon size={16} className="text-forge-crimson" />
          <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Engine Runtime Monitor</span>
        </div>
        <div className="flex gap-4">
          <button onClick={() => setIsPaused(!isPaused)} className="text-gray-500 hover:text-white transition-colors">
            {isPaused ? <Play size={14} /> : <Pause size={14} />}
          </button>
          <button onClick={() => setLogs([])} className="text-gray-500 hover:text-white transition-colors">
            <Trash2 size={14} />
          </button>
        </div>
      </div>
      
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-6 space-y-1 custom-scrollbar text-[11px] leading-relaxed"
      >
        {logs.map((log, i) => (
          <div key={i} className="flex gap-4 group">
            <span className="text-gray-600 shrink-0">[{log.timestamp}]</span>
            <span className={`shrink-0 w-12 ${getLevelStyle(log.level)}`}>{log.level}</span>
            <span className="text-gray-500 shrink-0">[{log.module}]</span>
            <span className="text-gray-300 break-all">{log.message}</span>
          </div>
        ))}
        {logs.length === 0 && (
          <div className="h-full flex items-center justify-center text-gray-700 italic">
            Awaiting engine heartbeat...
          </div>
        )}
      </div>

      <div className="bg-gray-900/50 px-6 py-2 border-t border-gray-800 flex justify-between items-center text-[9px] text-gray-600 font-bold uppercase">
        <span>Buffer: {logs.length}/500 lines</span>
        <span className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
          Live Stream Active
        </span>
      </div>
    </div>
  );
};
