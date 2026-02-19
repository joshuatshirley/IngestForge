"use client";

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  FileUp,
  Search,
  BookOpen,
  Settings,
  LogOut,
  ChevronLeft,
  Menu,
  Bot,
  BrainCircuit,
  Wand2,
  ShieldAlert,
  GraduationCap,
  Network,
  Compass,
  Hammer,
  Library,
  FolderOpen,
  ChevronDown,
  ChevronRight
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// US-1401.1 AC: Navigation - Library, Explorer, Mesh, Foundry
interface NavSection {
  name: string;
  icon: React.ElementType;
  items: { name: string; href: string; icon: React.ElementType }[];
}

const NAV_SECTIONS: NavSection[] = [
  {
    name: 'Library',
    icon: Library,
    items: [
      { name: 'Archive', href: '/library', icon: BookOpen },
      { name: 'Conflicts', href: '/library/conflicts', icon: ShieldAlert },
      { name: 'Transform', href: '/library/transform', icon: Wand2 },
    ],
  },
  {
    name: 'Explorer',
    icon: Compass,
    items: [
      { name: 'Research', href: '/research', icon: Search },
      { name: 'Discovery', href: '/research/discovery', icon: FolderOpen },
      { name: 'Memory', href: '/research/memory', icon: BrainCircuit },
    ],
  },
  {
    name: 'Mesh',
    icon: Network,
    items: [
      { name: 'Graph', href: '/research/graph', icon: Network },
      { name: 'Agents', href: '/research/agent', icon: Bot },
      { name: 'Apprentice', href: '/apprentice', icon: GraduationCap },
    ],
  },
  {
    name: 'Foundry',
    icon: Hammer,
    items: [
      { name: 'Dashboard', href: '/', icon: LayoutDashboard },
      { name: 'Ingestion', href: '/ingest', icon: FileUp },
      { name: 'Settings', href: '/settings', icon: Settings },
    ],
  },
];

// Flat nav items for mobile/collapsed view
const NAV_ITEMS = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Research', href: '/research', icon: Search },
  { name: 'Agents', href: '/research/agent', icon: Bot },
  { name: 'Memory', href: '/research/memory', icon: BrainCircuit },
  { name: 'Apprentice', href: '/apprentice', icon: GraduationCap },
  { name: 'Archive', href: '/library', icon: BookOpen },
  { name: 'Conflicts', href: '/library/conflicts', icon: ShieldAlert },
  { name: 'Transform', href: '/library/transform', icon: Wand2 },
  { name: 'Ingestion', href: '/ingest', icon: FileUp },
  { name: 'Settings', href: '/settings', icon: Settings },
];

// US-1401.1 AC: Responsive grid for 1080p and 4K displays
const SIDEBAR_WIDTHS = {
  collapsed: 'w-16 lg:w-20',
  expanded: 'w-56 lg:w-64 2xl:w-72',
};

export const Sidebar = () => {
  const pathname = usePathname();
  const [isCollapsed, setIsCollapsed] = React.useState(false);
  const [expandedSections, setExpandedSections] = React.useState<Set<string>>(
    () => new Set(['Library', 'Explorer', 'Mesh', 'Foundry'])
  );

  const toggleSection = (sectionName: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(sectionName)) {
        next.delete(sectionName);
      } else {
        next.add(sectionName);
      }
      return next;
    });
  };

  const isSectionActive = (section: NavSection) => {
    return section.items.some((item) => pathname === item.href);
  };

  // US-1401.1 AC: Collapsible sidebar with sections
  const renderExpandedNav = () => (
    <nav className="flex-1 px-3 space-y-1 mt-4 overflow-y-auto custom-scrollbar">
      {NAV_SECTIONS.map((section) => {
        const isExpanded = expandedSections.has(section.name);
        const sectionActive = isSectionActive(section);

        return (
          <div key={section.name} className="mb-2">
            <button
              onClick={() => toggleSection(section.name)}
              className={cn(
                "flex items-center justify-between w-full p-2.5 rounded-lg transition-all",
                "text-xs font-semibold uppercase tracking-wider",
                sectionActive
                  ? "text-forge-crimson bg-forge-crimson/10"
                  : "text-gray-500 hover:text-gray-300 hover:bg-gray-800/50"
              )}
            >
              <div className="flex items-center gap-2.5">
                <section.icon size={16} />
                <span>{section.name}</span>
              </div>
              {isExpanded ? (
                <ChevronDown size={14} />
              ) : (
                <ChevronRight size={14} />
              )}
            </button>

            {isExpanded && (
              <div className="mt-1 ml-2 space-y-0.5">
                {section.items.map((item) => {
                  const isActive = pathname === item.href;
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={cn(
                        "flex items-center gap-3 px-3 py-2 rounded-lg transition-all group",
                        "text-sm",
                        isActive
                          ? "bg-forge-crimson text-white shadow-md shadow-forge-crimson/20"
                          : "text-gray-400 hover:bg-gray-800 hover:text-white"
                      )}
                    >
                      <item.icon
                        size={18}
                        className={cn(
                          isActive ? "text-white" : "group-hover:text-forge-crimson"
                        )}
                      />
                      <span className="font-medium">{item.name}</span>
                    </Link>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </nav>
  );

  // Collapsed view shows only section icons
  const renderCollapsedNav = () => (
    <nav className="flex-1 px-2 space-y-2 mt-4">
      {NAV_SECTIONS.map((section) => {
        const sectionActive = isSectionActive(section);
        const firstItem = section.items[0];
        return (
          <Link
            key={section.name}
            href={firstItem.href}
            className={cn(
              "flex items-center justify-center p-3 rounded-xl transition-all group",
              sectionActive
                ? "bg-forge-crimson text-white shadow-lg"
                : "text-gray-400 hover:bg-gray-800 hover:text-white"
            )}
            title={section.name}
          >
            <section.icon
              size={22}
              className={cn(
                sectionActive ? "text-white" : "group-hover:text-forge-crimson"
              )}
            />
          </Link>
        );
      })}
    </nav>
  );

  return (
    <aside
      className={cn(
        // US-1401.1 AC: Foundry Dark theme - dark background with border
        "h-screen bg-forge-navy border-r border-gray-800/60",
        "transition-all duration-300 ease-in-out flex flex-col",
        // US-1401.1 AC: Responsive widths for 1080p and 4K
        isCollapsed ? SIDEBAR_WIDTHS.collapsed : SIDEBAR_WIDTHS.expanded
      )}
    >
      {/* Header with logo and toggle */}
      <div className="p-4 lg:p-6 flex items-center justify-between shrink-0">
        {!isCollapsed && (
          <span className="text-forge-crimson font-bold text-lg lg:text-xl tracking-tighter">
            INGESTFORGE
          </span>
        )}
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className={cn(
            "p-2 hover:bg-gray-800 rounded-lg transition-colors",
            isCollapsed && "mx-auto"
          )}
          aria-label="Toggle Sidebar"
        >
          {isCollapsed ? <Menu size={20} /> : <ChevronLeft size={20} />}
        </button>
      </div>

      {/* Navigation */}
      {isCollapsed ? renderCollapsedNav() : renderExpandedNav()}

      {/* Footer with logout */}
      <div className="p-3 lg:p-4 border-t border-gray-800/60 shrink-0">
        <button
          className={cn(
            "flex items-center gap-3 p-2.5 w-full rounded-lg",
            "text-gray-400 hover:text-forge-crimson hover:bg-gray-800/50 transition-all",
            isCollapsed && "justify-center"
          )}
        >
          <LogOut size={20} />
          {!isCollapsed && <span className="font-medium text-sm">Logout</span>}
        </button>
      </div>
    </aside>
  );
};
