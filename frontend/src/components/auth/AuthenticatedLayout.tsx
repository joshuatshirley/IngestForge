'use client';

import React, { useEffect } from 'react';
import { usePathname } from 'next/navigation';
import { Sidebar } from '@/components/Sidebar';
import { Header } from '@/components/Header';
import { ProtectedRoute } from './ProtectedRoute';
import { useAppDispatch, useAppSelector } from '@/hooks/redux';
import { initializeAuth, setCredentials } from '@/store/authSlice';

/**
 * Props for AuthenticatedLayout component.
 */
export interface AuthenticatedLayoutProps {
  /** Child components to render */
  children: React.ReactNode;
}

/**
 * Routes that do not require authentication.
 */
const PUBLIC_ROUTES = ['/login', '/signup', '/forgot-password', '/reset-password'];

/**
 * AuthenticatedLayout component that handles:
 * - Initial auth state hydration from localStorage
 * - Route-based layout (public vs protected)
 * - Wrapping protected routes with ProtectedRoute HOC
 */
export const AuthenticatedLayout: React.FC<AuthenticatedLayoutProps> = ({ children }) => {
  const pathname = usePathname();
  const dispatch = useAppDispatch();
  const isLoading = useAppSelector((state) => state.auth.isLoading);

  // Check if current route is public
  const isPublicRoute = PUBLIC_ROUTES.some((route) => pathname.startsWith(route));

  // Hydrate auth state from localStorage on mount
  useEffect(() => {
    const hydrateAuth = () => {
      try {
        const storedAuth = localStorage.getItem('ingestforge-auth');
        if (storedAuth) {
          const parsed = JSON.parse(storedAuth);
          // Check if token is not expired
          if (parsed.tokenExpiry && Date.now() < parsed.tokenExpiry) {
            dispatch(
              setCredentials({
                user: parsed.user,
                token: parsed.token,
                tokenExpiry: parsed.tokenExpiry,
              })
            );
          } else {
            // Token expired, clear storage
            localStorage.removeItem('ingestforge-auth');
            dispatch(initializeAuth());
          }
        } else {
          dispatch(initializeAuth());
        }
      } catch (error) {
        console.error('Failed to hydrate auth state:', error);
        dispatch(initializeAuth());
      }
    };

    hydrateAuth();
  }, [dispatch]);

  // Persist auth state to localStorage
  const token = useAppSelector((state) => state.auth.token);
  const user = useAppSelector((state) => state.auth.user);
  const tokenExpiry = useAppSelector((state) => state.auth.tokenExpiry);

  useEffect(() => {
    if (token && user) {
      localStorage.setItem(
        'ingestforge-auth',
        JSON.stringify({ token, user, tokenExpiry })
      );
    } else if (!isLoading) {
      localStorage.removeItem('ingestforge-auth');
    }
  }, [token, user, tokenExpiry, isLoading]);

  // Public routes - render without sidebar/header
  if (isPublicRoute) {
    return <>{children}</>;
  }

  // Protected routes - render with full layout and auth check
  return (
    <ProtectedRoute>
      <div className="flex h-screen w-screen">
        <Sidebar />
        <div className="flex flex-1 flex-col overflow-hidden">
          <Header />
          <main className="custom-scrollbar flex-1 overflow-y-auto p-8">{children}</main>
        </div>
      </div>
    </ProtectedRoute>
  );
};

export default AuthenticatedLayout;
