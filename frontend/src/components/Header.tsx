"use client";

import React from 'react';
import { Bell, User, Search } from 'lucide-react';

export const Header = () => {
  return (
    <header className="h-16 bg-forge-navy bg-opacity-50 backdrop-blur-md border-b border-gray-800 flex items-center justify-between px-8 sticky top-0 z-10">
      <div className="relative w-96">
        <span className="absolute inset-y-0 left-3 flex items-center text-gray-500">
          <Search size={18} />
        </span>
        <input 
          type="text"
          placeholder="Search knowledge base..."
          className="w-full bg-forge-blue border border-gray-700 rounded-full py-2 pl-10 pr-4 text-sm focus:outline-none focus:border-forge-crimson transition-colors"
        />
      </div>

      <div className="flex items-center gap-6">
        <button className="relative p-2 text-gray-400 hover:text-white transition-colors" aria-label="Notifications">
          <Bell size={20} />
          <span className="absolute top-2 right-2 w-2 h-2 bg-forge-crimson rounded-full"></span>
        </button>
        
        <div className="flex items-center gap-3 pl-6 border-l border-gray-800">
          <div className="text-right">
            <p className="text-sm font-semibold">Researcher</p>
            <p className="text-xs text-gray-500">Local Instance</p>
          </div>
          <div className="w-10 h-10 bg-forge-blue border border-gray-700 rounded-full flex items-center justify-center text-forge-crimson">
            <User size={24} />
          </div>
        </div>
      </div>
    </header>
  );
};
