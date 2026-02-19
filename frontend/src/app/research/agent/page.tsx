"use client";

import React, { useState } from 'react';
import { Bot, ShieldAlert, History } from 'lucide-react';
import { 
  useRunAgentMissionMutation, 
  useGetAgentStatusQuery,
  useGeneratePlanMutation,
  usePauseAgentMissionMutation,
  useResumeAgentMissionMutation
} from '@/store/api/ingestforgeApi';
import { useToast } from '@/components/ToastProvider';
import { ResearchRoadmap } from '@/components/research/ResearchRoadmap';
import { AgentMissionControl } from '@/components/chat/AgentMissionControl';
import { ReasoningChain } from '@/components/chat/ReasoningChain';
import { FinalSynthesis } from '@/components/chat/FinalSynthesis';
import { AgentJobStatus } from '@/components/chat/types';

export default function AgentPage() {
  const { showToast } = useToast();
  const [task, setTask] = useState('');
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [roadmap, setRoadmap] = useState<any>(null);
  
  const [generatePlan, { isLoading: isPlanning }] = useGeneratePlanMutation();
  const [runMission, { isLoading: isStarting }] = useRunAgentMissionMutation();
  const [pauseMission] = usePauseAgentMissionMutation();
  const [resumeMission] = useResumeAgentMissionMutation();
  
  const { data: jobStatus } = useGetAgentStatusQuery(currentJobId!, {
    skip: !currentJobId,
    pollingInterval: 3000,
  });

  const handlePause = async () => {
    if (!currentJobId) return;
    try {
      await pauseMission(currentJobId).unwrap();
      showToast('Mission Paused', 'info');
    } catch (err) {
      showToast('Failed to pause mission', 'error');
    }
  };

  const handleResume = async () => {
    if (!currentJobId) return;
    try {
      await resumeMission(currentJobId).unwrap();
      showToast('Mission Resumed', 'success');
    } catch (err) {
      showToast('Failed to resume mission', 'error');
    }
  };

  const handlePlanMission = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!task.trim()) return;
    
    try {
      const res = await generatePlan({ task, provider: 'llamacpp' }).unwrap();
      setRoadmap(res);
      showToast('Research Strategy Generated', 'success');
    } catch (err) {
      showToast('Strategy generation failed', 'error');
    }
  };

  const handleStartMission = async () => {
    try {
      const res = await runMission({ 
        task, 
        max_steps: 10, 
        provider: 'llamacpp',
        roadmap: roadmap?.tasks 
      }).unwrap();
      setCurrentJobId(res.job_id);
      showToast('Autonomous Mission Launched', 'success');
    } catch (err) {
      showToast('Failed to start agent mission', 'error');
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in duration-500 pb-20">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <Bot size={32} className="text-forge-crimson" />
            Agent Mission Control
          </h1>
          <p className="text-gray-400 mt-2">Deploy autonomous research agents to synthesize complex topics from your corpus.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Input Control */}
        <div className="lg:col-span-1 space-y-6">
          <AgentMissionControl 
            task={task}
            setTask={setTask}
            roadmap={roadmap}
            setRoadmap={setRoadmap}
            currentJobId={currentJobId}
            setCurrentJobId={setCurrentJobId}
            isPlanning={isPlanning}
            isStarting={isStarting}
            jobStatus={jobStatus}
            onPlan={handlePlanMission}
            onStart={handleStartMission}
          />

          {roadmap && !currentJobId && (
            <div className="forge-card p-8 border-forge-crimson/20 bg-gray-900/40">
              <ResearchRoadmap objective={task} tasks={roadmap.tasks} />
            </div>
          )}

          <div className="forge-card border-dashed border-2 border-gray-800 p-6 flex flex-col items-center text-center space-y-4 opacity-50">
            <ShieldAlert size={24} className="text-yellow-500" />
            <p className="text-[10px] font-bold text-gray-400 uppercase tracking-tighter leading-tight">
              Safety Protocol Active: Agent is restricted to local knowledge base and web search only.
            </p>
          </div>
        </div>

        {/* Live Mission Status */}
        <div className="lg:col-span-2 space-y-6">
          {!currentJobId ? (
            <div className="forge-card h-full min-h-[500px] flex flex-col items-center justify-center text-center space-y-6 border-gray-800 opacity-30">
              <History size={64} />
              <p className="text-sm italic">No active missions. Enter a research objective to begin.</p>
            </div>
          ) : (
            <div className="space-y-6">
              {jobStatus?.tasks && (
                <div className="forge-card p-8 border-forge-crimson/20 bg-gray-900/40">
                  <ResearchRoadmap objective={task} tasks={jobStatus.tasks} />
                </div>
              )}

              <ReasoningChain 
                jobId={currentJobId}
                status={jobStatus as AgentJobStatus}
                onPause={handlePause}
                onResume={handleResume}
              />

              {jobStatus?.answer && (
                <FinalSynthesis 
                  answer={jobStatus.answer}
                  verification={jobStatus.verification}
                />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
