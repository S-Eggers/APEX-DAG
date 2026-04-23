import React from 'react';
import { GraphMode, LabelOption } from '../../../types/GraphTypes';
import {
  PanelContainer,
  PanelHeader,
  PanelBody,
  PanelSection,
  PanelCodeBlock,
  PanelLabelSelect
} from './PanelComponents';

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
    <PanelContainer>
      <PanelHeader title={node.label || 'Node Details'} onClose={onClose} />

      <PanelBody>
        {mode === 'labeling' && (
          <PanelLabelSelect
            label="Node Domain Label"
            value={node.node_type}
            options={options}
            onChange={onChange}
          />
        )}

        <PanelSection title="Type ID">
          <p className="text-sm font-mono text-gray-800">
            {node.node_type !== undefined ? node.node_type : 'N/A'}
            {nodeTypeLabel ? ` (${nodeTypeLabel})` : ''}
          </p>
        </PanelSection>

        {node.base_inputs && (
          <PanelSection title="Parameters / Imports">
            <div className="bg-green-50 p-2 rounded text-xs text-green-800 border border-green-200 font-mono overflow-x-auto whitespace-pre-wrap">
              {node.base_inputs}
            </div>
          </PanelSection>
        )}

        {(mode === 'labeling' || mode === 'lineage') && (
          <PanelSection title="Transformation History">
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
          </PanelSection>
        )}

        <PanelSection title="Source Code">
          <PanelCodeBlock code={node.code} />
        </PanelSection>
      </PanelBody>
    </PanelContainer>
  );
}
