import React from 'react';
import { LabelOption } from '../../../types/GraphTypes';

interface EdgeDetailsPanelProps {
  edge: any;
  mode: string;
  options: LabelOption[];
  onChange: (newLabelValue: number) => void;
  onClose: () => void;
}

export default function EdgeDetailsPanel({
  edge,
  mode,
  options,
  onChange,
  onClose
}: EdgeDetailsPanelProps) {
  const data = typeof edge.data === 'function' ? edge.data() : edge;

  const getTaxonomyLabel = (typeNum: number) => {
    return `TYPE_${typeNum}`;
  };

  return (
    <div className="absolute top-0 right-0 w-96 h-full bg-white border-l border-gray-200 shadow-xl flex flex-col z-20 overflow-y-auto">
      <div className="p-4 border-b border-gray-100 flex justify-between items-center bg-gray-50">
        <h3 className="font-bold text-gray-800">Edge Details</h3>
        <a
          onClick={onClose}
          className="hover:cursor-pointer !text-gray-400 !hover:text-gray-800 !hover:bg-gray-200 !p-1 !rounded !transition-colors !focus:outline-none"
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

      <div className="p-5 flex flex-col gap-6">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest block mb-1">
              Structural Type
            </span>
            <span className="text-xs font-semibold text-blue-600 bg-blue-50 px-2 py-1 rounded">
              {getTaxonomyLabel(data.edge_type)}
            </span>
          </div>
          <div>
            <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest block mb-1">
              Operation
            </span>
            <span className="text-xs font-mono text-gray-800 bg-gray-100 px-2 py-1 rounded">
              {data.label || 'edge'}
            </span>
          </div>
        </div>

        <div>
          <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2">
            Source Code
          </label>
          <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto border border-gray-800 shadow-inner">
            <pre className="text-pink-400 font-mono text-xs leading-relaxed whitespace-pre-wrap">
              <code>
                {data.raw_code || data.label || 'No source available'}
              </code>
            </pre>
          </div>
        </div>

        {mode === 'labeling' && (
          <div className="pt-4 border-t border-gray-100">
            <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2">
              Domain Label
            </label>
            <select
              className="w-full p-2 border border-gray-200 rounded text-sm bg-white focus:ring-2 focus:ring-blue-500"
              value={data.predicted_label ?? ''}
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
        )}

        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-100">
          <div>
            <span className="text-[10px] font-bold text-gray-400 uppercase block mb-1">
              From
            </span>
            <div
              className="text-xs font-mono text-gray-600 bg-gray-50 p-2 rounded truncate"
              title={edge.source().data('label')}
            >
              {edge.source().data('label')}
            </div>
          </div>
          <div>
            <span className="text-[10px] font-bold text-gray-400 uppercase block mb-1">
              To
            </span>
            <div
              className="text-xs font-mono text-gray-600 bg-gray-50 p-2 rounded truncate"
              title={edge.target().data('label')}
            >
              {edge.target().data('label')}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
