export interface Message {
  role: 'user' | 'ai';
  text: string;
  results?: any[];
}

export interface SourcedResult {
  id: string;
  content: string;
  score: number;
  metadata: {
    source: string;
    bookmarked?: boolean;
    [key: string]: any;
  };
}

export interface AgentStep {
  thought: string;
  action?: string;
  observation?: string;
}

export interface AgentJobStatus {
  status: 'PENDING' | 'RUNNING' | 'PAUSED' | 'COMPLETED' | 'FAILED';
  steps: AgentStep[];
  tasks?: any[];
  answer?: string;
  verification?: {
    score: number;
    critic_notes: string;
    claims_verified: number;
  };
}
