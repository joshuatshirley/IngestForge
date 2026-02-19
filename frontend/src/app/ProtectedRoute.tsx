"use client";

import React, { useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { useSelector } from 'react-redux';
import { RootState } from '@/store';
import { Loader2 } from 'lucide-react';

export default function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const { isAuthenticated, token } = useSelector((state: RootState) => state.auth);

  useEffect(() => {
    if (!isAuthenticated && pathname !== '/login') {
      router.push('/login');
    }
  }, [isAuthenticated, router, pathname]);

  if (!isAuthenticated && pathname !== '/login') {
    return (
      <div className="h-screen w-screen bg-forge-navy flex flex-col items-center justify-center gap-4 text-gray-500">
        <Loader2 size={40} className="animate-spin text-forge-crimson" />
        <p className="text-sm font-medium animate-pulse uppercase tracking-widest">Validating Credentials...</p>
      </div>
    );
  }

  return <>{children}</>;
}
