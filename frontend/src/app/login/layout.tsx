import React from 'react';

/**
 * Login page layout - minimal layout without sidebar/header.
 */
export default function LoginLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return <>{children}</>;
}
