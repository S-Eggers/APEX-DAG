import React from 'react';
import { GraphMode, LabelOption } from '../../../types/GraphTypes';

interface NodeDetailsPanelProps {
  node: any;
  mode: GraphMode;
  options: LabelOption[];
  onChange: (newLabelValue: number) => void;
  onClose: () => void;
}

export default function NodeDetailsPanel({
  node,
  mode,
  options,
  onChange,
  onClose
}: NodeDetailsPanelProps) {
  const nodeTypeLabel = options.find(
    opt => opt.value === node.node_type
  )?.label;

  return (
    <div className="absolute top-0 right-0 w-80 h-full bg-white border-l border-gray-200 shadow-xl flex flex-col z-20 transform transition-transform duration-300">
      <div className="p-4 border-b border-gray-100 flex justify-between items-center bg-gray-50">
        <h3 className="font-bold text-gray-800 truncate" title={node.label}>
          {node.label || 'Node Details'}
        </h3>
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

      <div className="p-4 overflow-y-auto grow">
        {mode === 'labeling' && (
          <div className="mb-6 p-3 bg-blue-50 border border-blue-200 rounded">
            <label className="block text-xs font-bold text-blue-800 uppercase mb-2">
              Node Domain Label
            </label>
            <select
              className="w-full p-2 border rounded bg-white text-gray-800 focus:ring-2 focus:ring-blue-500"
              value={node.node_type ?? ''}
              onChange={e => onChange(Number(e.target.value))}
            >
              <option value="" disabled>
                Select a node label...
              </option>
              {options.map(opt => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="mb-6">
          <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">
            Type ID
          </span>
          <p className="text-sm font-mono text-gray-800 mt-1">
            {node.node_type !== undefined ? node.node_type : 'N/A'}
            {nodeTypeLabel ? ` (${nodeTypeLabel})` : ''}
          </p>
        </div>

        {node.base_inputs && (
          <div className="mb-6">
            <span className="text-xs font-bold text-gray-400 uppercase tracking-wider block mb-2">
              Parameters / Imports
            </span>
            <div className="bg-green-50 p-2 rounded text-xs text-green-800 border border-green-200 font-mono overflow-x-auto whitespace-pre-wrap">
              {node.base_inputs}
            </div>
          </div>
        )}

        <div className="mb-6">
          <span className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2 block">
            Transformation History
          </span>
          {node.transform_history && node.transform_history.length > 0 ? (
            <div className="space-y-3">
              {node.transform_history.map((step: any, idx: number) => (
                <div
                  key={idx}
                  className="bg-blue-50 p-3 rounded border border-blue-100 flex flex-col gap-2"
                >
                  <div className="flex justify-between items-center border-b border-blue-200 pb-1">
                    <span className="text-xs font-bold text-blue-800 uppercase truncate pr-2">
                      {step.operation}
                    </span>
                    <span className="text-xs text-blue-600 font-mono whitespace-nowrap">
                      → {step.target_node}
                    </span>
                  </div>
                  {step.transform_code && (
                    <div className="text-xs text-gray-700 font-mono bg-white p-2 rounded border border-gray-200 overflow-x-auto whitespace-pre-wrap">
                      {step.transform_code}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-gray-50 p-3 rounded text-xs border border-gray-200 border-dashed text-gray-500 italic">
              No linear transformations were contracted into this node.
            </div>
          )}
        </div>

        <div className="mb-6">
          <span className="text-xs font-bold text-gray-400 uppercase tracking-wider block mb-2">
            Original Code
          </span>
          {node.code ? (
            <pre className="bg-gray-100 p-2 rounded text-xs text-gray-700 overflow-x-auto whitespace-pre-wrap border border-gray-200">
              {node.code}
            </pre>
          ) : (
            <div className="bg-gray-50 p-3 rounded text-xs border border-gray-200 border-dashed text-gray-500 italic">
              Source code unavailable for this node.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
