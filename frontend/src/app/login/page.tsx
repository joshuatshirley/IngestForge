"use client";

import React from 'react';
import { useRouter } from 'next/navigation';
import { 
  ShieldCheck, 
  Lock, 
  Mail, 
  ArrowRight,
  Loader2
} from 'lucide-react';
import { useDispatch } from 'react-redux';
import { setCredentials } from '@/store/authSlice';
import { useToast } from '@/components/ToastProvider';

export default function LoginPage() {
  const router = useRouter();
  const dispatch = useDispatch();
  const { showToast } = useToast();
  const [loading, setLoading] = React.useState(false);
  const [formData, setFormData] = React.useState({
    email: '',
    password: ''
  });

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    // Simulate API call to /v1/auth/login
    setTimeout(() => {
      if (formData.password.length >= 8) {
        dispatch(setCredentials({
          user: { id: 'u1', name: 'Researcher', email: formData.email, role: 'admin' },
          token: 'mock_jwt_token_123'
        }));
        showToast('Login successful', 'success');
        router.push('/');
      } else {
        showToast('Invalid credentials (password must be 8+ chars)', 'error');
      }
      setLoading(false);
    }, 1000);
  };

  return (
    <div className="fixed inset-0 z-[200] bg-forge-navy flex items-center justify-center p-4 overflow-hidden">
      {/* Background Orbs */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-forge-crimson opacity-5 blur-[120px] rounded-full" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-600 opacity-5 blur-[120px] rounded-full" />

      <div className="w-full max-w-md space-y-8 animate-in fade-in zoom-in-95 duration-500">
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gray-800 rounded-3xl border border-gray-700 text-forge-crimson mb-4 shadow-2xl">
            <ShieldCheck size={32} />
          </div>
          <h1 className="text-3xl font-black tracking-tighter text-white">INGESTFORGE</h1>
          <p className="text-gray-500 text-sm font-medium uppercase tracking-widest">Research Portal v1.1</p>
        </div>

        <form onSubmit={handleLogin} className="forge-card p-8 space-y-6 border-gray-800 bg-opacity-40 backdrop-blur-xl shadow-2xl">
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest ml-1">Email Address</label>
              <div className="relative group">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 group-focus-within:text-forge-crimson transition-colors" size={18} />
                <input 
                  required
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({...formData, email: e.target.value})}
                  className="w-full bg-gray-900 border border-gray-800 rounded-2xl py-3 pl-12 pr-4 text-sm focus:outline-none focus:border-forge-crimson transition-all"
                  placeholder="name@institution.edu"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest ml-1">Password</label>
              <div className="relative group">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 group-focus-within:text-forge-crimson transition-colors" size={18} />
                <input 
                  required
                  type="password"
                  value={formData.password}
                  onChange={(e) => setFormData({...formData, password: e.target.value})}
                  className="w-full bg-gray-900 border border-gray-800 rounded-2xl py-3 pl-12 pr-4 text-sm focus:outline-none focus:border-forge-crimson transition-all"
                  placeholder="••••••••"
                />
              </div>
            </div>
          </div>

          <button 
            type="submit" 
            disabled={loading}
            className="w-full btn-primary py-4 rounded-2xl flex items-center justify-center gap-2 group shadow-xl shadow-forge-crimson/20 disabled:opacity-50"
          >
            {loading ? <Loader2 size={20} className="animate-spin" /> : (
              <>
                <span className="font-bold">Enter Laboratory</span>
                <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
              </>
            )}
          </button>

          <div className="pt-4 text-center">
            <p className="text-[10px] text-gray-600 font-medium">Local Instance • Authorized Personnel Only</p>
          </div>
        </form>
      </div>
    </div>
  );
}
