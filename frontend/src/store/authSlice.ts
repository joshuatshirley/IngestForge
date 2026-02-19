import { createSlice, PayloadAction } from '@reduxjs/toolkit';

/**
 * User role type for role-based access control.
 */
export type UserRole = 'admin' | 'researcher' | 'viewer' | 'guest';

/**
 * User interface representing authenticated user data.
 */
export interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
}

/**
 * Authentication state interface.
 */
export interface AuthState {
  user: User | null;
  token: string | null;
  tokenExpiry: number | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

/**
 * Payload for setting credentials.
 */
export interface SetCredentialsPayload {
  user: User;
  token: string;
  tokenExpiry?: number;
}

const initialState: AuthState = {
  user: null,
  token: null,
  tokenExpiry: null,
  isAuthenticated: false,
  isLoading: true, // Start with loading true for initial auth check
};

export const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    setCredentials: (state, action: PayloadAction<SetCredentialsPayload>) => {
      state.user = action.payload.user;
      state.token = action.payload.token;
      state.tokenExpiry = action.payload.tokenExpiry ?? null;
      state.isAuthenticated = true;
      state.isLoading = false;
    },
    logout: (state) => {
      state.user = null;
      state.token = null;
      state.tokenExpiry = null;
      state.isAuthenticated = false;
      state.isLoading = false;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    /**
     * Called after initial auth check completes (e.g., from localStorage/cookies).
     */
    initializeAuth: (state) => {
      state.isLoading = false;
    },
  },
});

export const { setCredentials, logout, setLoading, initializeAuth } = authSlice.actions;

/**
 * Selector to check if token is expired.
 */
export const selectIsTokenExpired = (state: { auth: AuthState }): boolean => {
  const { tokenExpiry } = state.auth;
  if (!tokenExpiry) {
    return false; // No expiry set means token is valid
  }
  return Date.now() > tokenExpiry;
};

/**
 * Selector to check if user has one of the required roles.
 */
export const selectHasRole = (
  state: { auth: AuthState },
  requiredRoles: UserRole[]
): boolean => {
  const { user } = state.auth;
  if (!user) {
    return false;
  }
  return requiredRoles.includes(user.role);
};

export default authSlice.reducer;
