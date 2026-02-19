/**
 * Nexus Management Service.
 * 
 * Task 271: Frontend API client for Workspace Nexus.
 */

import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export enum NexusStatus {
  ONLINE = 'ONLINE',
  OFFLINE = 'OFFLINE',
  REVOKED = 'REVOKED',
  PENDING = 'PENDING',
}

export interface NexusPeer {
  id: string;
  name: string;
  url: string;
  status: NexusStatus;
  last_seen?: string;
  failure_count: number;
}

export const nexusService = {
  /**
   * Fetch all registered peers.
   */
  async getPeers(): Promise<NexusPeer[]> {
    const response = await axios.get(`${API_BASE}/v1/nexus/peers`);
    return response.data;
  },

  /**
   * Register a new remote Nexus.
   */
  async registerPeer(peer: Partial<NexusPeer> & { api_key_hash: string }): Promise<void> {
    await axios.post(`${API_BASE}/v1/nexus/peers`, peer);
  },

  /**
   * Revoke access for a peer (Kill-Switch).
   */
  async revokePeer(peerId: string): Promise<void> {
    await axios.delete(`${API_BASE}/v1/nexus/peers/${peerId}`);
  },

  /**
   * Trigger manual health check.
   */
  async pingPeer(peerId: string): Promise<NexusStatus> {
    const response = await axios.post(`${API_BASE}/v1/nexus/peers/${peerId}/ping`);
    return response.data.new_status;
  },
};
