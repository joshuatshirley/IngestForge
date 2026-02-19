"use client";

import React, { useState } from 'react';
import { 
  Zap, Clock, Target, BarChart3, ChevronRight, Sparkles, Trophy, Loader2,
  BookOpen, FilePlus, BrainCircuit, GraduationCap
} from 'lucide-react';
import { useGetDueCardsQuery, useRateCardMutation, useGenerateStudyMaterialMutation, useGetStudyJobStatusQuery } from '@/store/api/ingestforgeApi';
import { useToast } from '@/components/ToastProvider';

export default function StudyPage() {
  const { showToast } = useToast();
  const [activeTab, setActiveTab] = useState<'review' | 'create'>('review');
  const [genType, setGenType] = useState<'glossary' | 'quiz' | 'notes'>('glossary');
  const [topic, setTopic] = useState('');
  const [genJobId, setGenJobId] = useState<string | null>(null);

  // Review Hooks
  const { data: dueData, isLoading: loadingDue, refetch } = useGetDueCardsQuery();
  const [rateCard] = useRateCardMutation();
  const [isStudying, setIsStudying] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);

  // Generation Hooks
  const [generateMaterial, { isLoading: isGenerating }] = useGenerateStudyMaterialMutation();
  const { data: genStatus } = useGetStudyJobStatusQuery(genJobId!, { skip: !genJobId });

  // --- Review Logic ---
  const currentCard = dueData?.cards[currentIndex];
  const handleRate = async (quality: number, label: string) => {
    if (!currentCard) return;
    await rateCard({ cardId: currentCard.id, quality });
    showToast(`Mastery: ${label}`, 'success');
    if (currentIndex + 1 < (dueData?.cards.length || 0)) {
      setCurrentIndex(currentIndex + 1); setIsFlipped(false);
    } else {
      setIsStudying(false); showToast('Session Complete!', 'success'); refetch();
    }
  };

  // --- Generation Logic ---
  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!topic) return;
    try {
      const res = await generateMaterial({ type: genType, topic }).unwrap();
      setGenJobId(res.job_id);
      showToast('Generation started...', 'info');
    } catch (err) { showToast('Generation failed', 'error'); }
  };

  // --- Review Mode UI ---
  if (isStudying && currentCard) {
    return (
      <div className="max-w-4xl mx-auto space-y-12 h-full flex flex-col items-center justify-center animate-in zoom-in-95 duration-300">
        <div className="w-full flex justify-between items-center px-4">
          <span className="text-[10px] font-bold text-gray-500 uppercase tracking-[0.2em]">Progress: {currentIndex + 1} / {dueData?.cards.length}</span>
          <button onClick={() => setIsStudying(false)} className="text-[10px] font-bold text-gray-500 hover:text-forge-crimson uppercase tracking-widest">Abort</button>
        </div>
        <div onClick={() => setIsFlipped(!isFlipped)} className={`relative w-full max-w-2xl aspect-[1.6/1] cursor-pointer perspective-1000 transition-all duration-500 transform-style-3d ${isFlipped ? 'rotate-y-180' : ''}`}>
          <div className="absolute inset-0 bg-forge-blue/40 backdrop-blur-xl border-2 border-gray-800 rounded-[3rem] p-12 flex flex-col items-center justify-center text-center backface-hidden shadow-2xl">
            <h2 className="text-3xl font-bold text-white">{currentCard.question}</h2>
            <div className="mt-12 flex items-center gap-2 text-[10px] text-gray-500 uppercase tracking-widest font-bold"><Sparkles size={12} /> Click to reveal</div>
          </div>
          <div className="absolute inset-0 bg-gray-900/90 backdrop-blur-2xl border-2 border-forge-accent/30 rounded-[3rem] p-12 flex flex-col items-center justify-center text-center backface-hidden rotate-y-180 shadow-2xl">
            <p className="text-xl text-gray-200">{currentCard.answer}</p>
          </div>
        </div>
        <div className={`flex gap-3 w-full max-w-2xl transition-all duration-500 ${isFlipped ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4 pointer-events-none'}`}>
          {[{l:'Again',v:1,c:'red'},{l:'Hard',v:3,c:'orange'},{l:'Good',v:4,c:'green'},{l:'Easy',v:5,c:'blue'}].map(b=>(
            <button key={b.l} onClick={()=>handleRate(b.v,b.l)} className={`flex-1 py-5 rounded-2xl border bg-gray-900/50 font-bold text-[10px] uppercase tracking-widest hover:scale-[1.02] text-${b.c}-400 border-${b.c}-900/30`}>{b.l}</button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in duration-500 pb-20">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <GraduationCap size={32} className="text-forge-crimson" />
            Study Factory
          </h1>
          <p className="text-gray-400 mt-2">Master your research with active recall and AI-generated materials.</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-8 border-b border-gray-800">
        <button onClick={() => setActiveTab('review')} className={`pb-4 text-sm font-bold uppercase tracking-widest transition-all relative ${activeTab === 'review' ? 'text-white' : 'text-gray-500'}`}>
          Review Session
          {activeTab === 'review' && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-forge-crimson" />}
        </button>
        <button onClick={() => setActiveTab('create')} className={`pb-4 text-sm font-bold uppercase tracking-widest transition-all relative ${activeTab === 'create' ? 'text-white' : 'text-gray-500'}`}>
          Generate Materials
          {activeTab === 'create' && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-forge-crimson" />}
        </button>
      </div>

      {activeTab === 'review' ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-in slide-in-from-left-4">
          <div className="md:col-span-2 forge-card p-8 border-gray-800 bg-gray-900/20 flex flex-col justify-between">
            <div>
              <h3 className="font-bold text-xl text-white mb-2">Ready for Review</h3>
              <p className="text-gray-400 text-sm">You have {dueData?.count || 0} cards scheduled for today based on SM-2 algorithm.</p>
            </div>
            <button onClick={() => setIsStudying(true)} disabled={!dueData?.count} className="mt-8 btn-primary px-8 py-4 rounded-xl flex items-center justify-center gap-2 shadow-lg disabled:opacity-50">
              <Sparkles size={18} /> <span className="font-bold uppercase tracking-widest">Start Session</span>
            </button>
          </div>
          <div className="forge-card flex flex-col justify-center items-center text-center p-8 border-gray-800 opacity-60">
            <Trophy size={48} className="text-yellow-500 mb-4" />
            <div className="text-2xl font-bold text-white">85%</div>
            <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Retention Rate</div>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 animate-in slide-in-from-right-4">
          <div className="forge-card p-8 bg-gradient-to-br from-forge-blue/10 to-transparent">
            <h3 className="font-bold mb-6 flex items-center gap-2 text-sm text-forge-accent">
              <BrainCircuit size={18} />
              Material Generator
            </h3>
            <form onSubmit={handleGenerate} className="space-y-6">
              <div className="grid grid-cols-3 gap-2">
                {[
                  { id: 'glossary', label: 'Glossary', icon: BookOpen },
                  { id: 'quiz', label: 'Quiz', icon: FilePlus },
                  { id: 'notes', label: 'Notes', icon: FileText }
                ].map((type) => (
                  <button type="button" key={type.id} onClick={() => setGenType(type.id as any)} className={`p-4 rounded-xl border flex flex-col items-center gap-2 transition-all ${genType === type.id ? 'bg-forge-crimson/20 border-forge-crimson text-white' : 'bg-gray-900 border-gray-800 text-gray-500 hover:border-gray-600'}`}>
                    <type.icon size={20} />
                    <span className="text-[10px] font-bold uppercase">{type.label}</span>
                  </button>
                ))}
              </div>
              <div className="space-y-2">
                <label className="text-[10px] font-black text-gray-500 uppercase tracking-widest">Source Topic</label>
                <input required value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="e.g. Quantum Mechanics" className="w-full bg-gray-900 border border-gray-800 rounded-xl px-4 py-3 text-sm focus:border-forge-crimson outline-none transition-all" />
              </div>
              <button type="submit" disabled={isGenerating} className="w-full btn-primary py-4 rounded-xl flex items-center justify-center gap-2 shadow-lg disabled:opacity-50">
                {isGenerating ? <Loader2 className="animate-spin" /> : <Zap size={18} />}
                <span className="font-bold uppercase tracking-widest">Generate</span>
              </button>
            </form>
          </div>

          <div className="forge-card border-dashed border-2 border-gray-800 p-8 min-h-[300px]">
            <h3 className="font-bold mb-4 text-xs uppercase tracking-widest text-gray-500">Output Console</h3>
            {genStatus?.status === 'RUNNING' && (
              <div className="flex flex-col items-center justify-center h-48 space-y-4 text-forge-crimson animate-pulse">
                <Loader2 size={32} className="animate-spin" />
                <span className="text-xs font-bold uppercase tracking-widest">Synthesizing...</span>
              </div>
            )}
            {genStatus?.status === 'COMPLETED' && (
              <div className="animate-in fade-in zoom-in-95">
                <div className="flex items-center gap-2 text-green-400 mb-4">
                  <CheckCircle2 size={16} /> <span className="text-xs font-bold uppercase">Ready</span>
                </div>
                <pre className="bg-gray-900 p-4 rounded-xl text-xs text-gray-300 overflow-auto max-h-64 border border-gray-800">
                  {JSON.stringify(genStatus.result, null, 2)}
                </pre>
              </div>
            )}
            {!genStatus && <div className="h-full flex items-center justify-center text-gray-600 text-xs italic">Waiting for input...</div>}
          </div>
        </div>
      )}
    </div>
  );
}
