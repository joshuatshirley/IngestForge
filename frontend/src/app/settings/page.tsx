"use client";

import React from 'react';
import { 
  Key, Database, Activity, ShieldCheck, HardDrive, RefreshCcw, CheckCircle, XCircle,
  Settings2, ChevronDown, ChevronUp, Terminal, Cpu, Zap
} from 'lucide-react';
import { useGetHealthQuery, useGetPacksQuery, useTogglePackMutation } from '@/store/api/ingestforgeApi';
import { useToast } from '@/components/ToastProvider';
import { LogTerminal } from '@/components/system/LogTerminal';
import { PackConfigForm } from '@/components/settings/PackConfigForm';
import { PerformanceMonitor } from '@/components/settings/PerformanceMonitor';
import { NexusPeerManager } from '@/components/nexus/NexusPeerManager';

export default function SettingsPage() {
  const { showToast } = useToast();
  const { data: health, isLoading, refetch } = useGetHealthQuery();
  const { data: packs } = useGetPacksQuery();
  const [togglePack] = useTogglePackMutation();
  const [showLogs, setShowLogs] = React.useState(false);
  const [showPerformance, setShowPerformance] = React.useState(false);
  const [activeConfigPack, setActiveConfigPack] = React.useState<string | null>(null);

  const handleTogglePack = async (id: string) => {
    try {
      await togglePack(id).unwrap();
      showToast(`Pack status updated`, 'success');
    } catch (err) {
      showToast('Failed to toggle pack', 'error');
    }
  };

  const handleTest = async (label: string) => {
    await refetch();
    showToast(`Diagnostics for ${label} completed`, 'info');
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in duration-500 pb-20">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">System Settings</h1>
          <p className="text-gray-400 mt-2">Manage API providers, storage backends, and hardware resources.</p>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={() => setShowPerformance(!showPerformance)}
            className={`flex items-center gap-2 text-xs px-4 py-2 rounded-xl border transition-all ${showPerformance ? 'bg-forge-accent border-forge-accent text-black font-black' : 'bg-gray-800 border-gray-700 text-gray-400'}`}
          >
            <Cpu size={14} />
            {showPerformance ? 'Hide Performance' : 'System Performance'}
          </button>
          <button 
            onClick={() => setShowLogs(!showLogs)}
            className={`flex items-center gap-2 text-xs px-4 py-2 rounded-xl border transition-all ${showLogs ? 'bg-forge-crimson border-forge-crimson text-white' : 'bg-gray-800 border-gray-700 text-gray-400'}`}
          >
            <Terminal size={14} />
            {showLogs ? 'Hide Logs' : 'View Logs'}
          </button>
        </div>
      </div>

      {showPerformance && (
        <div className="animate-in slide-in-from-top-4 duration-500">
          <PerformanceMonitor />
        </div>
      )}

      {showLogs && (
        <div className="animate-in slide-in-from-top-4 duration-500">
          <LogTerminal />
        </div>
      )}

      <div className="grid grid-cols-1 gap-8">
        {/* Forge Packs (Vertical Management) */}
        <div className="forge-card border-none bg-gradient-to-br from-forge-crimson/10 to-transparent">
          <h3 className="font-bold mb-6 flex items-center gap-2 text-white text-sm uppercase tracking-widest">
            <Zap size={18} className="text-forge-crimson" />
            Vertical Forge Packs
          </h3>
          <div className="space-y-4">
            {packs?.map((pack: any) => (
              <div key={pack.id} className="flex flex-col">
                <div className="flex items-center justify-between p-5 bg-gray-900/40 rounded-3xl border border-gray-800">
                  <div className="flex gap-4">
                    <div className={`p-3 rounded-2xl bg-gray-800 ${pack.active ? 'text-forge-crimson' : 'text-gray-600'}`}>
                      <Activity size={20} />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="font-bold text-sm text-gray-200">{pack.name}</p>
                        <span className="text-[8px] font-black bg-gray-800 px-1.5 py-0.5 rounded text-gray-500 uppercase tracking-tighter">v{pack.version}</span>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">{pack.description}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <button 
                      onClick={() => setActiveConfigPack(activeConfigPack === pack.id ? null : pack.id)}
                      className="p-2 bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-white rounded-xl border border-gray-700 transition-all"
                    >
                      {activeConfigPack === pack.id ? <ChevronUp size={14} /> : <Settings2 size={14} />}
                    </button>
                    <button 
                      onClick={() => handleTogglePack(pack.id)}
                      className={`px-6 py-2 rounded-xl text-[10px] font-bold uppercase tracking-widest transition-all ${
                        pack.active 
                          ? 'bg-forge-crimson text-white shadow-lg shadow-forge-crimson/20' 
                          : 'bg-gray-800 text-gray-500 border border-gray-700 hover:text-white'
                      }`}
                    >
                      {pack.active ? 'Active' : 'Enable'}
                    </button>
                  </div>
                </div>
                
                {activeConfigPack === pack.id && (
                  <div className="mt-4 p-8 bg-black/40 rounded-[2rem] border border-gray-800/50 animate-in slide-in-from-top-2 duration-300">
                    <PackConfigForm packId={pack.id} />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Workspace Nexus Management (Task 271) */}
        <div className="forge-card border-none bg-gradient-to-br from-cyan-900/20 to-transparent">
          <h3 className="font-bold mb-6 flex items-center gap-2 text-white text-sm uppercase tracking-widest">
            <Database size={18} className="text-cyan-400" />
            Workspace Nexus Federation
          </h3>
          <NexusPeerManager />
        </div>

        {/* Health Dashboard (Doctor Parity) */}
        <div className="forge-card border-none bg-gradient-to-br from-forge-blue/40 to-transparent">
          <h3 className="font-bold mb-6 flex items-center gap-2 text-white text-sm uppercase tracking-widest">
            <Activity size={18} className="text-forge-accent" />
            System Integrity
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {health?.checks.map((check: any) => (
              <div key={check.label} className="flex items-center justify-between p-5 bg-gray-900/50 rounded-3xl border border-gray-800 hover:border-gray-600 transition-all group">
                <div className="flex items-center gap-4">
                  <div className={`p-2 rounded-xl ${check.ok ? 'bg-green-900/20 text-green-500' : 'bg-red-900/20 text-red-500'}`}>
                    {check.ok ? <CheckCircle size={18} /> : <XCircle size={18} />}
                  </div>
                  <div>
                    <p className="text-sm font-bold text-gray-200">{check.label}</p>
                    <p className="text-[10px] text-gray-500 font-medium uppercase tracking-widest mt-0.5">{check.status}</p>
                  </div>
                </div>
                <button 
                  onClick={() => handleTest(check.label)}
                  className="opacity-0 group-hover:opacity-100 text-[10px] font-bold text-gray-500 hover:text-white uppercase tracking-tighter transition-all"
                >
                  Retest
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
