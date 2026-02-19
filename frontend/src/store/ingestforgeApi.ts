import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export interface SystemStatus {
  documents: number;
  mastery: string;
  load: string;
  backend: string;
}

export interface HealthCheck {
  status: string;
  engine: string;
}

export const ingestforgeApi = createApi({
  reducerPath: 'ingestforgeApi',
  baseQuery: fetchBaseQuery({ baseUrl: 'http://localhost:8000' }),
  tagTypes: ['Status', 'Ingest', 'Health'],
  endpoints: (builder) => ({
    getHealth: builder.query<HealthCheck, void>({
      query: () => '/health',
      providesTags: ['Health'],
    }),
    getStatus: builder.query<SystemStatus, void>({
      query: () => '/v1/status',
      providesTags: ['Status'],
    }),
    uploadFile: builder.mutation<{ id: string; status: string }, FormData>({
      query: (formData) => ({
        url: '/v1/ingest/upload',
        method: 'POST',
        body: formData,
      }),
      invalidatesTags: ['Status'],
    }),
  }),
});

export const { 
  useGetHealthQuery, 
  useGetStatusQuery, 
  useUploadFileMutation 
} = ingestforgeApi;
