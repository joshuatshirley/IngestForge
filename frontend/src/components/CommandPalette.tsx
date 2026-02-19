"use client";

import React, { useEffect, useState } from 'react';
import { 
  Search, 
  FileUp, 
  BookOpen, 
  Settings, 
  Terminal,
  Zap,
  Activity,
  Trash2,
  Command as CommandIcon
} from 'lucide-react';
import { useRouter } from 'next/navigation';

interface Command {
  id: string;
  name: string;
  icon: React.ReactNode;
  shortcut: string;
  action: () => void;
}

export const CommandPalette = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const router = useRouter();

  const commands: Command[] = [
    { id: 'ingest', name: 'Ingest Document', icon: <FileUp size={18} />, shortcut: 'I', action: () => router.push('/ingest') },
    { id: 'research', name: 'Start Research', icon: <Search size={18} />, shortcut: 'R', action: () => router.push('/research') },
    { id: 'study', name: 'Study Flashcards', icon: <BookOpen size={18} />, shortcut: 'S', action: () => router.push('/study') },
    { id: 'status', name: 'System Status', icon: <Zap size={18} />, shortcut: 'Z', action: () => router.push('/') },
    { id: 'doctor', name: 'Run Engine Doctor', icon: <Activity size={18} />, shortcut: 'D', action: () => router.push('/settings') },
    { id: 'cleanup', name: 'Cleanup Project', icon: <Trash2 size={18} />, shortcut: 'C', action: () => { router.push('/settings'); } },
    { id: 'settings', name: 'Open Settings', icon: <Settings size={18} />, shortcut: ',', action: () => router.push('/settings') },
  ];

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen((prev) => !prev);
      }
      if (e.key === 'Escape') setIsOpen(false);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh] px-4 bg-black bg-opacity-60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="w-full max-w-2xl bg-forge-blue border border-gray-700 rounded-2xl shadow-2xl overflow-hidden">
        <div className="flex items-center px-4 border-b border-gray-800">
          <Terminal size={20} className="text-gray-500" />
          <input 
            autoFocus
            type="text" 
            placeholder="Type a command or search..."
            className="w-full bg-transparent py-4 px-4 outline-none text-white placeholder-gray-600"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="flex items-center gap-1 bg-gray-800 px-2 py-1 rounded text-[10px] text-gray-400 font-bold">
            ESC
          </div>
        </div>

        <div className="p-2 max-h-96 overflow-y-auto">
          {commands
            .filter(c => c.name.toLowerCase().includes(query.toLowerCase()))
            .map((cmd) => (
              <button
                key={cmd.id}
                onClick={() => { cmd.action(); setIsOpen(false); }}
                className="w-full flex items-center justify-between p-3 hover:bg-gray-800 rounded-xl transition-colors group"
              >
                <div className="flex items-center gap-4 text-gray-300 group-hover:text-white">
                  <span className="text-forge-crimson">{cmd.icon}</span>
                  <span className="font-medium text-sm">{cmd.name}</span>
                </div>
                <div className="flex items-center gap-1 bg-gray-900 px-2 py-1 rounded text-[10px] text-gray-500 font-bold">
                  <CommandIcon size={10} /> {cmd.shortcut}
                </div>
              </button>
            ))}
        </div>

        <div className="bg-gray-900 p-3 flex justify-between items-center text-[10px] text-gray-500 font-bold uppercase tracking-widest px-6">
          <span>{commands.length} Commands available</span>
          <span className="flex items-center gap-2 italic">
            Command Aliasing Active
          </span>
        </div>
      </div>
    </div>
  );
};
