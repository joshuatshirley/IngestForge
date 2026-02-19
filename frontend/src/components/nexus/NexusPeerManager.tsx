'use client';

/**
 * Nexus Peer Management Dashboard.
 * 
 * Task 271: Secure UI for federated peer lifecycle.
 */

import React, { useState, useEffect } from 'react';
import { nexusService, NexusPeer, NexusStatus } from '@/services/nexusService';

export const NexusPeerManager: React.FC = () => {
  const [peers, setPeers] = useState<NexusPeer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPeers = async () => {
    try {
      setLoading(true);
      const data = await nexusService.getPeers();
      setPeers(data);
      setError(null);
    } catch (err) {
      setError('Failed to load Nexus peers.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPeers();
    // Poll for health updates every 30s (Rule #2 compliant)
    const interval = setInterval(fetchPeers, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleRevoke = async (id: string) => {
    if (!confirm('Are you sure you want to REVOKE all access for this peer?')) return;
    try {
      await nexusService.revokePeer(id);
      fetchPeers();
    } catch (err) {
      alert('Revocation failed.');
    }
  };

  const handlePing = async (id: string) => {
    try {
      await nexusService.pingPeer(id);
      fetchPeers();
    } catch (err) {
      alert('Ping failed.');
    }
  };

  const getStatusColor = (status: NexusStatus) => {
    switch (status) {
      case NexusStatus.ONLINE: return 'text-green-400 bg-green-900/20';
      case NexusStatus.OFFLINE: return 'text-red-400 bg-red-900/20';
      case NexusStatus.REVOKED: return 'text-gray-400 bg-gray-900/20';
      default: return 'text-yellow-400 bg-yellow-900/20';
    }
  };

  return (
    <div className="flex flex-col gap-6 p-6 bg-forge-navy rounded-lg border border-gray-800">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold text-white">Workspace Nexus Registry</h2>
        <button 
          onClick={fetchPeers}
          className="px-4 py-2 text-sm bg-gray-800 hover:bg-gray-700 text-gray-200 rounded transition-colors"
        >
          Refresh
        </button>
      </div>

      {error && <div className="p-3 text-red-400 bg-red-900/10 border border-red-900/50 rounded">{error}</div>}

      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm text-gray-300">
          <thead className="bg-forge-black text-gray-400 uppercase text-xs">
            <tr>
              <th className="px-4 py-3 font-medium">Name</th>
              <th className="px-4 py-3 font-medium">Status</th>
              <th className="px-4 py-3 font-medium">Last Seen</th>
              <th className="px-4 py-3 font-medium text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {peers.length === 0 && !loading && (
              <tr><td colSpan={4} className="px-4 py-8 text-center text-gray-500 italic">No peers registered.</td></tr>
            )}
            {peers.map((peer) => (
              <tr key={peer.id} className="hover:bg-forge-black/50 transition-colors">
                <td className="px-4 py-4 font-medium text-white">
                  {peer.name}
                  <div className="text-xs text-gray-500 font-mono mt-1">{peer.url}</div>
                </td>
                <td className="px-4 py-4">
                  <span className={`px-2 py-1 rounded text-xs font-bold ${getStatusColor(peer.status)}`}>
                    {peer.status}
                  </span>
                </td>
                <td className="px-4 py-4 text-gray-400">
                  {peer.last_seen ? new Date(peer.last_seen).toLocaleString() : 'Never'}
                </td>
                <td className="px-4 py-4 text-right flex justify-end gap-2">
                  <button 
                    onClick={() => handlePing(peer.id)}
                    className="p-2 hover:text-cyan-400 transition-colors"
                    title="Ping Now"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </button>
                  <button 
                    onClick={() => handleRevoke(peer.id)}
                    className="p-2 text-red-500 hover:text-red-300 transition-colors"
                    title="Revoke Global Access"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728L5.636 5.636" />
                    </svg>
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
