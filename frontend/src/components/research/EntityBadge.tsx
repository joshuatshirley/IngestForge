/**
 * EntityBadge Component
 * US-152: SRC-UI Enhanced entity badges.
 */

import React from 'react';
import { User, Building2, MapPin, Calendar, Tag } from 'lucide-react';

export type EntityType = 'person' | 'organization' | 'location' | 'date' | 'concept' | string;

interface EntityBadgeProps {
  type: EntityType;
  text: string;
  active?: boolean;
}

const ENTITY_CONFIG: Record<string, { icon: any, color: string, bg: string }> = {
  person: { icon: User, color: 'text-blue-400', bg: 'bg-blue-400/10 border-blue-400/20' },
  organization: { icon: Building2, color: 'text-green-400', bg: 'bg-green-400/10 border-green-400/20' },
  location: { icon: MapPin, color: 'text-purple-400', bg: 'bg-purple-400/10 border-purple-400/20' },
  date: { icon: Calendar, color: 'text-orange-400', bg: 'bg-orange-400/10 border-orange-400/20' },
  concept: { icon: Tag, color: 'text-forge-accent', bg: 'bg-forge-accent/10 border-forge-accent/20' },
};

export const EntityBadge: React.FC<EntityBadgeProps> = ({ type, text, active }) => {
  const config = ENTITY_CONFIG[type.toLowerCase()] || ENTITY_CONFIG.concept;
  const Icon = config.icon;

  return (
    <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border transition-all ${
      active 
        ? 'bg-forge-cyan border-forge-cyan text-black scale-110 shadow-lg shadow-forge-cyan/20' 
        : `${config.bg} ${config.color} hover:scale-105`
    } text-[10px] font-bold uppercase tracking-tight`}>
      <Icon size={10} />
      <span>{text}</span>
    </div>
  );
};
