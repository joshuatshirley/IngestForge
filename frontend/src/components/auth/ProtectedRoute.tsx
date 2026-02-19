'use client';

import React, { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAppSelector } from '@/hooks/redux';
import { selectIsTokenExpired, selectHasRole, type UserRole } from '@/store/authSlice';

/**
 * Props interface for ProtectedRoute component.
 */
export interface ProtectedRouteProps {
  /** Child components to render when authenticated */
  children: React.ReactNode;
  /** Optional array of roles required to access this route */
  requiredRoles?: UserRole[];
  /** Custom loading component (optional) */
  loadingComponent?: React.ReactNode;
  /** Custom unauthorized component (optional) */
  unauthorizedComponent?: React.ReactNode;
  /** Redirect path when unauthenticated (default: /login) */
  redirectTo?: string;
}

/**
 * Default loading spinner component.
 */
const DefaultLoadingComponent = () => (
  <div className="flex h-screen w-full items-center justify-center bg-forge-navy">
    <div className="flex flex-col items-center gap-4">
      <div className="h-12 w-12 animate-spin rounded-full border-4 border-gray-700 border-t-forge-crimson" />
      <p className="text-sm text-gray-400">Verifying authentication...</p>
    </div>
  </div>
);

/**
 * Default unauthorized component shown when user lacks required role.
 */
const DefaultUnauthorizedComponent = () => (
  <div className="flex h-screen w-full items-center justify-center bg-forge-navy">
    <div className="flex flex-col items-center gap-4 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-red-900 bg-opacity-30">
        <svg
          className="h-8 w-8 text-red-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
      </div>
      <h2 className="text-xl font-bold text-white">Access Denied</h2>
      <p className="max-w-sm text-sm text-gray-400">
        You do not have permission to access this resource. Please contact an administrator if you
        believe this is an error.
      </p>
    </div>
  </div>
);

/**
 * ProtectedRoute HOC component that guards routes requiring authentication.
 *
 * Features:
 * - Checks for valid authentication token in Redux store
 * - Supports token expiry validation
 * - Role-based access control with requiredRoles prop
 * - Configurable redirect path
 * - Loading state while checking authentication
 *
 * @example
 * // Basic usage - requires authentication
 * <ProtectedRoute>
 *   <DashboardPage />
 * </ProtectedRoute>
 *
 * @example
 * // With role-based access
 * <ProtectedRoute requiredRoles={['admin', 'researcher']}>
 *   <AdminPanel />
 * </ProtectedRoute>
 *
 * @example
 * // With custom loading and redirect
 * <ProtectedRoute
 *   loadingComponent={<MyCustomLoader />}
 *   redirectTo="/auth/signin"
 * >
 *   <ProtectedContent />
 * </ProtectedRoute>
 */
export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredRoles,
  loadingComponent,
  unauthorizedComponent,
  redirectTo = '/login',
}) => {
  const router = useRouter();

  // Auth state selectors
  const token = useAppSelector((state) => state.auth.token);
  const isAuthenticated = useAppSelector((state) => state.auth.isAuthenticated);
  const isLoading = useAppSelector((state) => state.auth.isLoading);
  const isTokenExpired = useAppSelector(selectIsTokenExpired);
  const hasRequiredRole = useAppSelector((state) =>
    requiredRoles ? selectHasRole(state, requiredRoles) : true
  );

  // Determine if authentication is valid
  const isValidAuth = isAuthenticated && token && !isTokenExpired;

  useEffect(() => {
    // Skip redirect while still loading
    if (isLoading) {
      return;
    }

    // Redirect if not authenticated or token expired
    if (!isValidAuth) {
      // Encode current path for redirect after login
      const currentPath = window.location.pathname;
      const redirectUrl = currentPath !== '/' ? `${redirectTo}?returnTo=${encodeURIComponent(currentPath)}` : redirectTo;
      router.replace(redirectUrl);
    }
  }, [isLoading, isValidAuth, redirectTo, router]);

  // Show loading state while checking authentication
  if (isLoading) {
    return <>{loadingComponent ?? <DefaultLoadingComponent />}</>;
  }

  // Not authenticated - show nothing while redirecting
  if (!isValidAuth) {
    return <>{loadingComponent ?? <DefaultLoadingComponent />}</>;
  }

  // Authenticated but lacks required role
  if (requiredRoles && !hasRequiredRole) {
    return <>{unauthorizedComponent ?? <DefaultUnauthorizedComponent />}</>;
  }

  // Authenticated with proper role - render children
  return <>{children}</>;
};

export default ProtectedRoute;
