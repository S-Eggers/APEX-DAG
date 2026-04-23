import React, { ReactNode } from 'react';
import { LabelOption } from '../../../types/GraphTypes';

export function PanelContainer({ children }: { children: ReactNode }) {
  return (
    <div className="absolute top-0 right-0 w-96 h-full bg-white border-l border-gray-200 shadow-2xl flex flex-col z-20 transform transition-transform duration-300">
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
    <div className="p-4 border-b border-gray-100 flex justify-between items-center bg-gray-50 shrink-0">
      <h3 className="font-bold text-gray-800 truncate pr-4" title={title}>
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
      <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest block mb-2">
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
      <div className="bg-gray-50 p-3 rounded text-xs border border-gray-200 border-dashed text-gray-500 italic">
        {fallback}
      </div>
    );
  }
  return (
    <div className="bg-slate-50 rounded-md p-3 overflow-x-auto border border-slate-200 shadow-inner custom-scrollbar">
      <pre className="text-slate-700 font-mono text-xs leading-relaxed whitespace-pre-wrap">
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
    <div className="p-3 bg-blue-50 border border-blue-200 rounded-md">
      <label className="block text-xs font-bold text-blue-800 uppercase tracking-wider mb-2">
        {label}
      </label>
      <select
        className="w-full p-2 border border-gray-200 rounded text-sm bg-white focus:ring-2 focus:ring-blue-500"
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
          ? 'font-mono text-gray-800 bg-gray-100'
          : 'font-semibold text-blue-600 bg-blue-50'
      }`}
      title={text}
    >
      {text}
    </div>
  );
}
