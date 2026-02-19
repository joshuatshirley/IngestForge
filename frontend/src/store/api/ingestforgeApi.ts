/**
 * IngestForge RTK Query API Slice
 *
 * Provides typed API endpoints for the IngestForge Research Portal.
 */

import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const ingestforgeApi = createApi({
    reducerPath: 'ingestforgeApi',
    baseQuery: fetchBaseQuery({
        baseUrl: 'http://localhost:8000',
        prepareHeaders: (headers) => {
            headers.set('Accept', 'application/json');
            return headers;
        },
    }),
    tagTypes: ['Status', 'Health', 'Search', 'Job', 'Agent', 'Learning', 'Graph'],
    endpoints: (builder) => ({
        getStatus: builder.query<any, void>({
            query: () => '/v1/status',
            providesTags: ['Status'],
        }),
        getHealth: builder.query<any, void>({
            query: () => '/v1/health',
            providesTags: ['Health'],
        }),
        getSystemTelemetry: builder.query<any, void>({
            query: () => '/v1/system/telemetry',
            providesTags: ['Health'],
        }),
        searchCorpus: builder.mutation<any, { 
            query: string; 
            top_k: number; 
            library?: string; 
            filters?: any; 
            sort_by?: string;
            broadcast?: boolean;
            nexus_ids?: string[];
        }>({
            query: (params) => ({
                url: '/v1/search',
                method: 'POST',
                body: params,
            }),
        }),
        chat: builder.mutation<any, { 
            messages: any[];
            broadcast?: boolean;
            nexus_ids?: string[];
        }>({
            query: (body) => ({
                url: '/v1/chat',
                method: 'POST',
                body,
            }),
        }),
        addVerifiedExample: builder.mutation<any, any>({
            query: (example) => ({
                url: '/v1/learning/examples',
                method: 'POST',
                body: example,
            }),
        }),
        uploadFile: builder.mutation<any, File>({
            query: (file) => {
                const formData = new FormData();
                formData.append('file', file);
                return { url: '/v1/ingest/upload', method: 'POST', body: formData };
            },
            invalidatesTags: ['Status'],
        }),
        ingestRemoteDocument: builder.mutation<any, { platform: string; source_id: string; token?: string }>({
            query: (body) => ({ url: '/v1/ingest/remote', method: 'POST', body }),
            invalidatesTags: ['Status'],
        }),
        getJobStatus: builder.query<any, string>({
            query: (jobId) => `/v1/ingest/status/${jobId}`,
            providesTags: (result) => result ? [{ type: 'Job', id: result.id }] : ['Job'],
        }),
        
        // Agent Endpoints
        runAgentMission: builder.mutation<{ job_id: string }, { task: string; max_steps?: number; provider?: string; roadmap?: any[] }>({
            query: (body) => ({ url: '/v1/agent/run', method: 'POST', body }),
            invalidatesTags: ['Status'],
        }),
        generatePlan: builder.mutation<any, { task: string; provider?: string }>({
            query: (body) => ({ url: '/v1/agent/plan', method: 'POST', body }),
        }),
        getAgentStatus: builder.query<any, string>({
            query: (jobId) => `/v1/agent/status/${jobId}`,
            providesTags: (result) => result ? [{ type: 'Agent', id: result.id }] : ['Agent'],
        }),
        getAgentMemory: builder.query<{ facts: any[] }, void>({
            query: () => '/v1/agent/memory',
            providesTags: ['Agent'],
        }),
        deleteAgentMemory: builder.mutation<{ status: string }, number>({
            query: (id) => ({ url: `/v1/agent/memory/${id}`, method: 'DELETE' }),
            invalidatesTags: ['Agent'],
        }),
        pauseAgentMission: builder.mutation<{ status: string }, string>({
            query: (id) => ({ url: `/v1/agent/mission/${id}/pause`, method: 'POST' }),
            invalidatesTags: ['Agent'],
        }),
        resumeAgentMission: builder.mutation<{ status: string }, string>({
            query: (id) => ({ url: `/v1/agent/mission/${id}/resume`, method: 'POST' }),
            invalidatesTags: ['Agent'],
        }),

        // Corpus Management
        getLibraries: builder.query<{ libraries: any[] }, void>({
            query: () => '/v1/libraries',
            providesTags: ['Status'],
        }),
        createLibrary: builder.mutation<{ id: string }, { name: string; description?: string }>({
            query: (body) => ({ url: '/v1/libraries', method: 'POST', body }),
            invalidatesTags: ['Status'],
        }),
        deleteLibrary: builder.mutation<{ status: string }, string>({
            query: (id) => ({ url: `/v1/libraries/${id}`, method: 'DELETE' }),
            invalidatesTags: ['Status'],
        }),
        pushSync: builder.mutation<{ job_id: string; status: string }, void>({
            query: () => ({ url: '/v1/sync/push', method: 'POST' }),
            invalidatesTags: ['Status'],
        }),
        transformCorpus: builder.mutation<{ job_id: string }, { operation: string; target_library?: string; params?: any }>({
            query: (body) => ({ url: '/v1/corpus/transform', method: 'POST', body }),
            invalidatesTags: ['Status'],
        }),

        getDocuments: builder.query<any, void>({
            query: () => '/v1/library/documents',
            providesTags: ['Status'],
        }),
        getChunkContext: builder.query<any, string>({
            query: (id) => `/v1/library/context/${id}`,
        }),
        getTags: builder.query<any, void>({
            query: () => '/v1/library/tags',
            providesTags: ['Status'],
        }),
        getBookmarks: builder.query<any, void>({
            query: () => '/v1/library/bookmarks',
            providesTags: ['Search'],
        }),
        toggleBookmark: builder.mutation<any, string>({
            query: (id) => ({
                url: `/v1/library/bookmark/${id}`,
                method: 'POST',
            }),
            invalidatesTags: ['Search'],
        }),
        addTag: builder.mutation<any, { chunkId: string; tag: string }>({
            query: (body) => ({
                url: `/v1/library/tag`,
                method: 'POST',
                body,
            }),
            invalidatesTags: ['Search'],
        }),
        
        // Study Hub
        getDueCards: builder.query<any, void>({
            query: () => '/v1/study/due',
            providesTags: ['Job'],
        }),
        rateCard: builder.mutation<any, { cardId: string; quality: number }>({
            query: (body) => ({ url: '/v1/study/rate', method: 'POST', body }),
        }),
        generateStudyMaterial: builder.mutation<{ job_id: string }, { type: string; topic: string; count?: number; provider?: string }>({
            query: (body) => ({ url: '/v1/study/generate', method: 'POST', body }),
        }),
        getStudyJobStatus: builder.query<any, string>({
            query: (jobId) => `/v1/study/job/${jobId}`,
        }),

        // Analysis
        getContradictions: builder.query<{ conflicts: any[] }, void>({
            query: () => '/v1/analysis/contradictions',
            providesTags: ['Search'],
        }),
        getLiteraryAnalysis: builder.query<any, string>({
            query: (id) => `/v1/analysis/literary/${id}`,
            providesTags: (result, error, id) => [{ type: 'Search', id }],
        }),
        getTimeline: builder.query<any, { library?: string; document_id?: string }>({
            query: (params) => ({
                url: '/v1/analysis/timeline',
                params,
            }),
            providesTags: ['Search'],
        }),

        // Visuals
        getDashboardAnalytics: builder.query<any, void>({
            query: () => '/v1/analytics/dashboard',
        }),
        getGraph: builder.query<any, void>({
            query: () => '/v1/viz/graph',
        }),

        // US-1402.1: Enhanced Knowledge Mesh
        getKnowledgeMesh: builder.query<any, {
            max_nodes?: number;
            min_citations?: number;
            depth?: number;
            entity_types?: string;
            include_chunks?: boolean;
        }>({
            query: (params) => ({
                url: '/v1/viz/graph/knowledge-mesh',
                params,
            }),
            providesTags: ['Graph'],
        }),
        getPackConfig: builder.query<{ values: any; schema: any }, string>({
            query: (id) => `/v1/packs/${id}/config`,
            providesTags: (result, error, id) => [{ type: 'Status', id }],
        }),
        updatePackConfig: builder.mutation<any, { id: string; settings: any }>({
            query: ({ id, settings }) => ({ url: `/v1/packs/${id}/config`, method: 'POST', body: settings }),
            invalidatesTags: ['Status'],
        }),
        getPacks: builder.query<any[], void>({
            query: () => '/v1/packs',
            providesTags: ['Status'],
        }),
        togglePack: builder.mutation<any, string>({
            query: (id) => ({ url: `/v1/packs/${id}/toggle`, method: 'POST' }),
            invalidatesTags: ['Status'],
        }),

        // Learning Governance (US-2701.4)
        getLearningExamples: builder.query<any, { vertical_id?: string; entity_type?: string; status?: string; page?: number; page_size?: number }>({
            query: (params) => ({
                url: '/v1/learning/examples',
                params,
            }),
            providesTags: ['Learning'],
        }),
        getLearningStats: builder.query<any, void>({
            query: () => '/v1/learning/stats',
            providesTags: ['Learning'],
        }),
        approveLearningExample: builder.mutation<any, { example_id: string; approved_by?: string }>({
            query: (body) => ({ url: '/v1/learning/approve', method: 'POST', body }),
            invalidatesTags: ['Learning'],
        }),
        rejectLearningExample: builder.mutation<any, { example_id: string; reason?: string }>({
            query: (body) => ({ url: '/v1/learning/reject', method: 'POST', body }),
            invalidatesTags: ['Learning'],
        }),
        editLearningExample: builder.mutation<any, { example_id: string; entities?: any[]; entity_type?: string }>({
            query: (body) => ({ url: '/v1/learning/edit', method: 'PUT', body }),
            invalidatesTags: ['Learning'],
        }),
    }),
});

export const {
    useGetStatusQuery,
    useGetHealthQuery,
    useGetSystemTelemetryQuery,
    useSearchCorpusMutation,
    useUploadFileMutation,
    useIngestRemoteDocumentMutation,
    useGetJobStatusQuery,
    useRunAgentMissionMutation,
    useGeneratePlanMutation,
    useGetAgentStatusQuery,
    useGetAgentMemoryQuery,
    useDeleteAgentMemoryMutation,
    usePauseAgentMissionMutation,
    useResumeAgentMissionMutation,
    useGetLibrariesQuery,
    useCreateLibraryMutation,
    useDeleteLibraryMutation,
    usePushSyncMutation,
    useTransformCorpusMutation,
    useGetDocumentsQuery,
    useGetChunkContextQuery,
    useGetTagsQuery,
    useGetBookmarksQuery,
    useToggleBookmarkMutation,
    useAddTagMutation,
    useGetDueCardsQuery,
    useRateCardMutation,
    useGenerateStudyMaterialMutation,
    useGetStudyJobStatusQuery,
    useGetContradictionsQuery,
    useGetLiteraryAnalysisQuery,
    useGetDashboardAnalyticsQuery,
    useGetGraphQuery,
    useGetPackConfigQuery,
    useUpdatePackConfigMutation,
    useGetPacksQuery,
    useTogglePackMutation,
    // Learning Governance (US-2701.4)
    useGetLearningExamplesQuery,
    useGetLearningStatsQuery,
    useApproveLearningExampleMutation,
    useRejectLearningExampleMutation,
    useEditLearningExampleMutation,
    // US-1402.1: Knowledge Mesh
    useGetKnowledgeMeshQuery,
} = ingestforgeApi;
