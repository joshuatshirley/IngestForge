"use client";

import React from 'react';
import { 
  Activity, Files, Cpu, Zap, RefreshCcw, ShieldCheck, Search, Trophy, Sparkles
} from 'lucide-react';
import { 
  useGetStatusQuery, 
  useGetHealthQuery, 
  useGetDashboardAnalyticsQuery 
} from '@/store/api/ingestforgeApi';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell 
} from 'recharts';

export default function Dashboard() {
  const { data: status, isLoading: statusLoading, refetch: refetchStatus } = useGetStatusQuery();
  const { data: health } = useGetHealthQuery();
  const { data: analytics } = useGetDashboardAnalyticsQuery();

  const stats = [
    { label: 'Documents', value: status?.stats.totalDocuments?.toLocaleString() || '0', icon: Files, color: 'text-blue-400' },
    { label: 'Chunks', value: status?.stats.totalChunks?.toLocaleString() || '0', icon: Zap, color: 'text-yellow-400' },
    { label: 'Engine', value: status?.status || 'Offline', icon: Cpu, color: status?.status === 'online' ? 'text-green-400' : 'text-red-400' },
    { label: 'Health', value: health?.healthy ? 'Stable' : 'Issues', icon: ShieldCheck, color: health?.healthy ? 'text-green-400' : 'text-yellow-400' },
  ];

  const COLORS = ['#e94560', '#4fc3f7', '#4ecca3', '#9c27b0'];

  return (
    <div className="space-y-8 animate-in fade-in duration-500 pb-12">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">Research Dashboard</h1>
          <p className="text-gray-400 mt-2">Real-time intelligence monitor for your knowledge base.</p>
        </div>
        <button onClick={() => refetchStatus()} className="flex items-center gap-2 text-xs bg-gray-800 hover:bg-gray-700 px-4 py-2 rounded-xl transition-all border border-gray-700">
          <RefreshCcw size={14} className={statusLoading ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => (
          <div key={stat.label} className="forge-card flex items-center gap-4 group hover:bg-forge-blue/20 transition-all border-gray-800">
            <div className={`p-3 rounded-2xl bg-gray-800/50 border border-gray-700 ${stat.color}`}>
              <stat.icon size={24} />
            </div>
            <div>
              <p className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">{stat.label}</p>
              <p className="text-2xl font-bold text-white">{stat.value}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Mastery Over Time Chart */}
        <div className="lg:col-span-2 forge-card p-8 min-h-[400px] border-gray-800 bg-gray-900/20">
          <h3 className="font-bold mb-8 flex items-center gap-2 text-sm text-white">
            <Activity size={18} className="text-forge-crimson" />
            Concept Mastery Growth
          </h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={analytics?.mastery_over_time}>
                <defs>
                  <linearGradient id="colorMastery" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#e94560" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#e94560" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="date" stroke="#4b5563" fontSize={10} tickLine={false} axisLine={false} />
                <YAxis stroke="#4b5563" fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#16213e', border: '1px solid #374151', borderRadius: '12px', fontSize: '12px' }}
                  itemStyle={{ color: '#e94560' }}
                />
                <Area type="monotone" dataKey="mastery" stroke="#e94560" strokeWidth={3} fillOpacity={1} fill="url(#colorMastery)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Category Distribution Pie */}
        <div className="forge-card p-8 border-gray-800 bg-gray-900/20">
          <h3 className="font-bold mb-8 flex items-center gap-2 text-sm text-white">
            <Trophy size={18} className="text-yellow-500" />
            Topic Distribution
          </h3>
          <div className="h-[240px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={analytics?.category_distribution}
                  cx="50%" cy="50%" innerRadius={60} outerRadius={80}
                  paddingAngle={5} dataKey="value"
                >
                  {analytics?.category_distribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="none" />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-6 space-y-2">
            {analytics?.category_distribution.map((item, i) => (
              <div key={item.name} className="flex justify-between items-center text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                  <span className="text-gray-400">{item.name}</span>
                </div>
                <span className="text-gray-200 font-bold">{((item.value / 1000) * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="forge-card p-8 border-gray-800 bg-gradient-to-br from-forge-blue/20 to-transparent">
          <div className="flex items-center justify-between mb-8">
            <h3 className="font-bold flex items-center gap-2 text-sm text-white">
              <RefreshCcw size={18} className="text-forge-accent" />
              Engine Metadata
            </h3>
            <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest bg-gray-800 px-3 py-1 rounded-full">v1.1.0-stable</span>
          </div>
          <div className="grid grid-cols-2 gap-4 text-xs">
            <div className="p-4 bg-gray-900/40 rounded-2xl border border-gray-800">
              <p className="text-gray-500 mb-1">Processing Mode</p>
              <p className="font-bold text-gray-200">Balanced (CPU-only)</p>
            </div>
            <div className="p-4 bg-gray-900/40 rounded-2xl border border-gray-800">
              <p className="text-gray-500 mb-1">Parallel Workers</p>
              <p className="font-bold text-gray-200">4 Instances</p>
            </div>
          </div>
        </div>

        <div className="forge-card border-dashed border-2 border-gray-800 flex items-center justify-between p-10 group hover:border-forge-crimson/30 transition-all">
          <div className="flex items-center gap-6">
            <div className="w-14 h-14 bg-gray-800/50 rounded-2xl flex items-center justify-center text-forge-crimson border border-gray-700">
              <Zap size={28} />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white tracking-tight">Generate Study Materials</h3>
              <p className="text-xs text-gray-500 mt-1">Transform recent research into a new quiz.</p>
            </div>
          </div>
          <button className="flex items-center gap-2 text-[10px] font-bold text-white bg-forge-crimson px-6 py-3 rounded-xl shadow-xl shadow-forge-crimson/20 hover:scale-105 transition-all uppercase tracking-widest">
            <Sparkles size={14} />
            Auto-Build
          </button>
        </div>
      </div>
    </div>
  );
}
