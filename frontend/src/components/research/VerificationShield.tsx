"use client";

import React from 'react';
import { ShieldCheck, ShieldAlert, ShieldX, Info } from 'lucide-react';

interface VerificationProps {
  score: number;
  notes?: string;
  count?: number;
}

export const VerificationShield = ({ score, notes, count }: VerificationProps) => {
  // Map score to status
  let StatusIcon = ShieldCheck;
  let colorClass = "text-green-500 bg-green-500/10 border-green-500/20";
  let label = "Verified";

  if (score < 0.6) {
    StatusIcon = ShieldX;
    colorClass = "text-red-500 bg-red-500/10 border-red-500/20";
    label = "Contradicted";
  } else if (score < 0.85) {
    StatusIcon = ShieldAlert;
    colorClass = "text-yellow-500 bg-yellow-500/10 border-yellow-500/20";
    label = "Mixed Evidence";
  }

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border ${colorClass} animate-in fade-in zoom-in-95 duration-500 group relative cursor-help`}>
      <StatusIcon size={14} className={score >= 0.85 ? "animate-pulse" : ""} />
      <span className="text-[10px] font-black uppercase tracking-tighter">{label}</span>
      <span className="text-[10px] font-bold opacity-60">{(score * 100).toFixed(0)}%</span>

      {/* Tooltip */}
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-4 bg-gray-900 border border-gray-800 rounded-2xl shadow-2xl opacity-0 group-hover:opacity-100 pointer-events-none transition-all z-50">
        <div className="flex items-center gap-2 mb-2 border-b border-gray-800 pb-2">
          <Info size={12} className="text-forge-crimson" />
          <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Critic Audit Notes</span>
        </div>
        <p className="text-xs text-gray-300 leading-relaxed italic">"{notes || 'No critical notes available for this synthesis.'}"</p>
        <div className="mt-3 flex justify-between items-center text-[9px] font-bold text-gray-500 uppercase">
          <span>Claims Audited: {count || 0}</span>
          <span className="text-forge-accent">High Fidelity</span>
        </div>
      </div>
    </div>
  );
};
