import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';
import type { RootState, AppDispatch } from '@/store';

/**
 * Typed dispatch hook for Redux store.
 * Use this instead of plain `useDispatch` for type safety.
 */
export const useAppDispatch = () => useDispatch<AppDispatch>();

/**
 * Typed selector hook for Redux store.
 * Use this instead of plain `useSelector` for type safety.
 */
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
