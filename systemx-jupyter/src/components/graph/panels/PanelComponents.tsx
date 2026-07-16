import React, { ReactNode, useEffect } from 'react';
import { LabelOption, LeakageGoldEntry } from '../../../types/GraphTypes';

export function PanelContainer({
  children,
  onClose,
  ariaLabel = 'Details panel'
}: {
  children: ReactNode;
  onClose?: () => void;
  ariaLabel?: string;
}) {
  useEffect(() => {
    if (!onClose) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  return (
    <div
      role="complementary"
      aria-label={ariaLabel}
      className="absolute top-0 right-0 w-96 h-full bg-white dark:bg-[#161b22] border-l border-gray-200 dark:border-gray-700 shadow-2xl flex flex-col z-20 transform transition-transform duration-300"
    >
      {children}
    </div>
  );
}

export function PanelHeader({
  title,
  onClose
}: {
  title: string;
  onClose: () => void;
}) {
  return (
    <div className="p-4 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center bg-gray-50 dark:bg-[#21262d] shrink-0">
      <h3 className="font-bold text-gray-800 dark:text-gray-100 truncate pr-4" title={title}>
        {title}
      </h3>
      <a
        onClick={onClose}
        className="hover:cursor-pointer !text-gray-400 !hover:text-gray-800 !hover:bg-gray-200 !p-1 !rounded !transition-colors !focus:outline-none shrink-0"
        aria-label="Close panel"
      >
        <svg
          className="w-5 h-5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      </a>
    </div>
  );
}

export function PanelBody({ children }: { children: ReactNode }) {
  return (
    <div className="p-5 flex flex-col gap-6 overflow-y-auto grow">
      {children}
    </div>
  );
}

export function PanelSection({
  title,
  children
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div>
      <span className="text-[10px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-widest block mb-2">
        {title}
      </span>
      {children}
    </div>
  );
}

export function PanelCodeBlock({
  code,
  fallback = 'No source available'
}: {
  code?: string;
  fallback?: string;
}) {
  if (!code) {
    return (
      <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded text-xs border border-gray-200 dark:border-gray-600 border-dashed text-gray-500 dark:text-gray-400 italic">
        {fallback}
      </div>
    );
  }
  return (
    <div className="bg-slate-50 dark:bg-gray-800 rounded-md p-3 overflow-x-auto border border-slate-200 dark:border-gray-600 shadow-inner custom-scrollbar">
      <pre className="text-slate-700 dark:text-slate-300 font-mono text-xs leading-relaxed whitespace-pre-wrap">
        <code>{code}</code>
      </pre>
    </div>
  );
}

export function PanelLabelSelect({
  label,
  value,
  options,
  onChange
}: {
  label: string;
  value: number | string;
  options: LabelOption[];
  onChange: (val: number) => void;
}) {
  return (
    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 rounded-md">
      <label className="block text-xs font-bold text-blue-800 dark:text-blue-300 uppercase tracking-wider mb-2">
        {label}
      </label>
      <select
        aria-label={label}
        className="w-full p-2 border border-gray-200 dark:border-gray-600 rounded text-sm bg-white dark:bg-[#161b22] dark:text-gray-200 focus:ring-2 focus:ring-blue-500"
        value={value ?? ''}
        onChange={e => onChange(Number(e.target.value))}
      >
        <option value="" disabled>
          Select Label...
        </option>
        {options.map(opt => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export function LeakageGoldSelect({
  entries,
  value,
  onChange
}: {
  entries: LeakageGoldEntry[];
  value: number;
  onChange: (val: number) => void;
}) {
  const groups: { category: string; items: { index: number; label: string }[] }[] =
    [];
  entries.forEach((entry, index) => {
    let group = groups.find(g => g.category === entry.category);
    if (!group) {
      group = { category: entry.category, items: [] };
      groups.push(group);
    }
    group.items.push({ index, label: entry.label });
  });

  const selected = entries[value] ?? entries[0];

  if (!selected) return null;

  return (
    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 rounded-md">
      <label className="block text-xs font-bold text-blue-800 dark:text-blue-300 uppercase tracking-wider mb-2">
        Gold Leakage Label (human)
      </label>
      <div className="flex items-center gap-2">
        <span
          className="w-3 h-3 rounded-sm border border-black/10 dark:border-white/20 shrink-0"
          style={{ backgroundColor: selected.color }}
          aria-hidden="true"
        />
        <select
          aria-label="Gold Leakage Label"
          className="w-full p-2 border border-gray-200 dark:border-gray-600 rounded text-sm bg-white dark:bg-[#161b22] dark:text-gray-200 focus:ring-2 focus:ring-blue-500"
          value={value}
          onChange={e => onChange(Number(e.target.value))}
        >
          {groups.map(group => (
            <optgroup key={group.category} label={group.category}>
              {group.items.map(item => (
                <option key={item.index} value={item.index}>
                  {item.label}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </div>
      <p className="mt-2 text-xs text-gray-600 dark:text-gray-400">
        {selected.description}
      </p>
    </div>
  );
}

export function PanelBadge({
  text,
  isMono = false
}: {
  text: string;
  isMono?: boolean;
}) {
  return (
    <div
      className={`text-xs px-2 py-1 rounded truncate w-fit ${
        isMono
          ? 'font-mono text-gray-800 dark:text-gray-200 bg-gray-100 dark:bg-gray-700'
          : 'font-semibold text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30'
      }`}
      title={text}
    >
      {text}
    </div>
  );
}
