"use client";

import React from 'react';
import { 
  Cpu, 
  Database, 
  Zap, 
  Leaf, 
  Scale, 
  Rocket,
  ShieldCheck,
  RefreshCcw,
  Loader2
} from 'lucide-react';
import { useGetSystemTelemetryQuery, useUpdatePackConfigMutation } from '@/store/api/ingestforgeApi';
import { useToast } from '@/components/ToastProvider';

export const PerformanceMonitor = () => {
  const { showToast } = useToast();
  const { data: telemetry, isLoading, refetch } = useGetSystemTelemetryQuery(undefined, {
    pollingInterval: 5000,
  });
  const [updateConfig, { isLoading: isUpdating }] = useUpdatePackConfigMutation();

  const handleSetPreset = async (preset: string) => {
    try {
      // In a real implementation, this would update a global performance_mode setting
      await updateConfig({ id: 'performance', settings: { mode: preset } }).unwrap();
      showToast(`Performance mode set to: ${preset.toUpperCase()}`, 'success');
    } catch (err) {
      showToast('Failed to update preset', 'error');
    }
  };

  if (isLoading) return <div className="h-48 flex items-center justify-center"><Loader2 className="animate-spin text-forge-crimson" /></div>;

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      {/* Real-time Telemetry */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="forge-card p-6 border-gray-800 bg-gray-900/40">
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center gap-2 text-blue-400">
              <Database size={16} />
              <span className="text-[10px] font-black uppercase tracking-widest">Memory (RAM)</span>
            </div>
            <span className="text-xs font-bold text-gray-200">{telemetry?.ram.usage_percent}%</span>
          </div>
          <div className="w-full bg-gray-800 h-2 rounded-full overflow-hidden shadow-inner">
            <div 
              className={`h-full transition-all duration-1000 ${telemetry?.ram.usage_percent > 85 ? 'bg-red-500' : 'bg-blue-500'}`}
              style={{ width: `${telemetry?.ram.usage_percent}%` }}
            />
          </div>
          <p className="text-[9px] text-gray-500 mt-3 font-bold uppercase">{telemetry?.ram.available}GB Available / {telemetry?.ram.total}GB Total</p>
        </div>

        <div className="forge-card p-6 border-gray-800 bg-gray-900/40">
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center gap-2 text-forge-accent">
              <Cpu size={16} />
              <span className="text-[10px] font-black uppercase tracking-widest">Processor (CPU)</span>
            </div>
            <span className="text-xs font-bold text-gray-200">{telemetry?.cpu.usage_percent}%</span>
          </div>
          <div className="w-full bg-gray-800 h-2 rounded-full overflow-hidden shadow-inner">
            <div 
              className={`h-full transition-all duration-1000 ${telemetry?.cpu.usage_percent > 80 ? 'bg-orange-500' : 'bg-forge-accent'}`}
              style={{ width: `${telemetry?.cpu.usage_percent}%` }}
            />
          </div>
          <p className="text-[9px] text-gray-500 mt-3 font-bold uppercase">{telemetry?.cpu.count} Logical Cores Detected</p>
        </div>

        <div className="forge-card p-6 border-forge-crimson/20 bg-forge-crimson/5 flex flex-col justify-between">
          <div className="flex items-center gap-2 text-forge-crimson">
            <ShieldCheck size={16} />
            <span className="text-[10px] font-black uppercase tracking-widest">Optimal Config</span>
          </div>
          <div>
            <p className="text-lg font-bold text-white tracking-tight capitalize">{telemetry?.recommendation} Mode</p>
            <p className="text-[9px] text-gray-500 font-bold uppercase mt-1 leading-relaxed">
              Based on your hardware, we recommend the {telemetry?.recommendation} preset for stability.
            </p>
          </div>
        </div>
      </div>

      {/* Preset Selection */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { id: 'eco', name: 'Eco Mode', desc: 'Minimal RAM usage. Best for laptops/low-end devices.', icon: Leaf, color: 'text-green-400' },
          { id: 'balanced', name: 'Balanced', desc: 'Optimal trade-off between speed and resource usage.', icon: Scale, color: 'text-blue-400' },
          { id: 'performance', name: 'Performance', desc: 'High concurrency. Maximize use of available CPU/RAM.', icon: Rocket, color: 'text-forge-crimson' },
        ].map((p) => (
          <button 
            key={p.id}
            onClick={() => handleSetPreset(p.id)}
            className="forge-card p-6 text-left group hover:bg-gray-800/50 transition-all border-gray-800"
          >
            <div className={`p-3 rounded-xl bg-gray-900 border border-gray-800 mb-4 w-fit ${p.color} group-hover:scale-110 transition-transform`}>
              <p.icon size={20} />
            </div>
            <h4 className="font-bold text-sm text-white mb-1">{p.name}</h4>
            <p className="text-[10px] text-gray-500 leading-relaxed font-medium uppercase tracking-tighter">{p.desc}</p>
          </button>
        ))}
      </div>
    </div>
  );
};
