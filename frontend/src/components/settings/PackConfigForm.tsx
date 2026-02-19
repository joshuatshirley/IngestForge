"use client";

import React from 'react';
import { useGetPackConfigQuery, useUpdatePackConfigMutation } from '@/store/api/ingestforgeApi';
import { Save, Loader2, Info } from 'lucide-react';
import { useToast } from '@/components/ToastProvider';

interface FormProps {
  packId: string;
}

export const PackConfigForm = ({ packId }: FormProps) => {
  const { showToast } = useToast();
  const { data, isLoading } = useGetPackConfigQuery(packId);
  const [updateConfig, { isLoading: isUpdating }] = useUpdatePackConfigMutation();
  const [formData, setFormData] = React.useState<any>({});

  React.useEffect(() => {
    if (data?.values) setFormData(data.values);
  }, [data]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await updateConfig({ id: packId, settings: formData }).unwrap();
      showToast(`${packId.toUpperCase()} Settings Saved`, 'success');
    } catch (err) {
      showToast('Failed to save settings', 'error');
    }
  };

  if (isLoading) return <div className="p-12 flex justify-center"><Loader2 className="animate-spin text-forge-crimson" /></div>;

  const properties = data?.schema?.properties || {};

  return (
    <form onSubmit={handleSubmit} className="space-y-6 animate-in fade-in slide-in-from-top-2 duration-500">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {Object.keys(properties).map((key) => {
          const prop = properties[key];
          if (key === 'enabled' || key === 'version') return null; // Handled by toggle/static

          return (
            <div key={key} className="space-y-2">
              <label className="text-[10px] font-black text-gray-500 uppercase tracking-widest flex items-center gap-2">
                {key.replace(/_/g, ' ')}
                <Info size={10} className="text-forge-accent cursor-help" title={prop.description} />
              </label>
              
              {prop.type === 'boolean' ? (
                <div 
                  onClick={() => setFormData({...formData, [key]: !formData[key]})}
                  className={`w-12 h-6 rounded-full p-1 cursor-pointer transition-all ${formData[key] ? 'bg-forge-crimson' : 'bg-gray-800'}`}
                >
                  <div className={`w-4 h-4 bg-white rounded-full transition-transform ${formData[key] ? 'translate-x-6' : 'translate-x-0'}`} />
                </div>
              ) : prop.type === 'integer' ? (
                <input 
                  type="number"
                  value={formData[key] || 0}
                  onChange={(e) => setFormData({...formData, [key]: parseInt(e.target.value)})}
                  className="w-full bg-gray-900 border border-gray-800 rounded-xl px-4 py-2 text-sm focus:border-forge-crimson outline-none"
                />
              ) : (
                <input 
                  type="text"
                  value={formData[key] || ''}
                  onChange={(e) => setFormData({...formData, [key]: e.target.value})}
                  className="w-full bg-gray-900 border border-gray-800 rounded-xl px-4 py-2 text-sm focus:border-forge-crimson outline-none"
                />
              )}
            </div>
          );
        })}
      </div>

      <div className="pt-6 border-t border-gray-800 flex justify-end">
        <button 
          type="submit" 
          disabled={isUpdating}
          className="btn-primary px-8 py-2.5 rounded-xl flex items-center gap-2 text-xs font-bold uppercase tracking-widest shadow-xl shadow-forge-crimson/20 disabled:opacity-50"
        >
          {isUpdating ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
          Save Configuration
        </button>
      </div>
    </form>
  );
};
