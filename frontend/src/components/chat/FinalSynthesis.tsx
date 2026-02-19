"use client";

import React from 'react';
import { CheckCircle2, FileText } from 'lucide-react';
import { VerificationShield } from '@/components/research/VerificationShield';

interface FinalSynthesisProps {
  answer: string;
  verification?: {
    score: number;
    critic_notes: string;
    claims_verified: number;
  };
}

export const FinalSynthesis: React.FC<FinalSynthesisProps> = ({ answer, verification }) => {
  return (
    <div className="forge-card p-8 border-green-900/30 bg-green-900/5 animate-in zoom-in-95 duration-500">
      <div className="flex justify-between items-center mb-6">
        <h3 className="font-bold flex items-center gap-2 text-sm text-green-400">
          <CheckCircle2 size={18} />
          Final Synthesis
        </h3>
        {verification && (
          <VerificationShield 
            score={verification.score} 
            notes={verification.critic_notes}
            count={verification.claims_verified}
          />
        )}
      </div>
      <div className="prose prose-invert prose-sm max-w-none text-gray-200 leading-loose">
        {answer}
      </div>
      <div className="mt-8 pt-6 border-t border-green-900/20 flex gap-4">
        <button className="flex items-center gap-2 text-[10px] font-bold text-green-400 hover:text-white transition-colors uppercase">
          <FileText size={14} />
          Export Report
        </button>
      </div>
    </div>
  );
};
