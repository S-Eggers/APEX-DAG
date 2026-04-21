import React from 'react';
import { LabelOption } from '../../../types/GraphTypes';

interface EdgeAnnotationPanelProps {
  edge: any;
  options: LabelOption[];
  onChange: (newLabelValue: number) => void;
  onClose: () => void;
}

export default function EdgeAnnotationPanel({
  edge,
  options,
  onChange,
  onClose
}: EdgeAnnotationPanelProps) {
  return (
    <div className="absolute top-0 right-0 w-80 h-full bg-white border-l border-gray-200 shadow-xl flex flex-col z-20 transform transition-transform duration-300">
      <div className="p-4 border-b border-gray-100 flex justify-between items-center bg-gray-50">
        <h3 className="font-bold text-gray-800">Edit Edge Label</h3>
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

      <div className="p-4">
        <label className="block text-xs font-bold text-gray-500 uppercase mb-2">
          Domain Label
        </label>
        <select
          className="w-full p-2 border rounded bg-white text-gray-800 focus:ring-2 focus:ring-blue-500"
          value={edge.data('predicted_label') ?? ''}
          onChange={e => onChange(Number(e.target.value))}
        >
          <option value="" disabled>
            Select a label...
          </option>
          {options.map(opt => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>

        <div className="mt-6">
          <span className="text-xs text-gray-500 block mb-1">Source Node:</span>
          <code className="text-xs bg-gray-100 p-1 rounded">
            {edge.source().data('label')}
          </code>

          <span className="text-xs text-gray-500 block mb-1 mt-3">
            Target Node:
          </span>
          <code className="text-xs bg-gray-100 p-1 rounded">
            {edge.target().data('label')}
          </code>
        </div>
      </div>
    </div>
  );
}
