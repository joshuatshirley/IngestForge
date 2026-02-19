"use client";

import React from 'react';
import { Loader2 } from 'lucide-react';
import { Message } from './types';

interface MessageListProps {
  messages: Message[];
  isSearching: boolean;
}

export const MessageList: React.FC<MessageListProps> = ({ messages, isSearching }) => {
  return (
    <div className="flex-1 p-8 space-y-8 overflow-y-auto custom-scrollbar">
      {messages.map((msg, i) => (
        <div key={i} className={`flex gap-6 max-w-4xl ${msg.role === 'user' ? 'ml-auto flex-row-reverse' : ''}`}>
          <div className={`w-10 h-10 rounded-2xl flex-shrink-0 flex items-center justify-center font-bold text-sm ${msg.role === 'ai' ? 'bg-forge-crimson shadow-lg' : 'bg-blue-600 shadow-lg'}`}>
            {msg.role === 'ai' ? 'IF' : 'R'}
          </div>
          <div className={`bg-opacity-40 border border-gray-800 rounded-3xl p-6 text-gray-200 shadow-xl ${msg.role === 'ai' ? 'bg-forge-blue rounded-tl-none' : 'bg-forge-crimson !bg-opacity-100 rounded-tr-none text-white'}`}>
            <p>{msg.text}</p>
          </div>
        </div>
      ))}
      {isSearching && (
        <div className="flex gap-6 animate-pulse">
          <div className="w-10 h-10 bg-gray-800 rounded-2xl flex items-center justify-center">
            <Loader2 size={20} className="animate-spin" />
          </div>
          <div className="bg-gray-800 bg-opacity-40 border border-gray-800 rounded-3xl w-64 h-16" />
        </div>
      )}
    </div>
  );
};
