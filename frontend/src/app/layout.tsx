import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "IngestForge | Academic Research Portal",
  description: "Advanced RAG ecosystem for mission-critical research.",
};

import StoreProvider from "./StoreProvider";
import { ToastProvider } from "@/components/ToastProvider";
import { CommandPalette } from "@/components/CommandPalette";
import ProtectedRoute from "./ProtectedRoute";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-forge-navy text-white overflow-hidden`}>
        <StoreProvider>
          <ToastProvider>
            <ProtectedRoute>
              <div className="flex h-screen w-screen">
                <Sidebar />
                <div className="flex-1 flex flex-col overflow-hidden">
                  <Header />
                  <main className="flex-1 overflow-y-auto p-8 custom-scrollbar">
                    {children}
                  </main>
                </div>
              </div>
              <CommandPalette />
            </ProtectedRoute>
          </ToastProvider>
        </StoreProvider>
      </body>
    </html>
  );
}
