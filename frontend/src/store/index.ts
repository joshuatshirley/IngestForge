import { configureStore } from '@reduxjs/toolkit';
import uiReducer from './uiSlice';
import authReducer from './authSlice';
import graphReducer from './slices/graphSlice';
import searchReducer from './slices/searchSlice';
import { ingestforgeApi } from './api/ingestforgeApi';

export const store = configureStore({
  reducer: {
    ui: uiReducer,
    auth: authReducer,
    graph: graphReducer,
    search: searchReducer,
    [ingestforgeApi.reducerPath]: ingestforgeApi.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(ingestforgeApi.middleware),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Re-export API hooks for convenience
export {
  useGetStatusQuery,
  useGetHealthQuery,
  useSearchQuery,
  useUploadFileMutation,
  useGetJobStatusQuery,
} from './api/ingestforgeApi';
