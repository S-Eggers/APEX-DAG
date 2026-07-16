import React from 'react';
import {
  GraphMode,
  LabelOption,
  LeakageGoldEntry,
  GraphNodeData
} from '../../../types/GraphTypes';
import {
  PanelContainer,
  PanelHeader,
  PanelBody,
  PanelSection,
  PanelCodeBlock,
  PanelLabelSelect,
  LeakageGoldSelect,
  PanelBadge
} from './PanelComponents';
import FeatureImportanceSection from './FeatureImportanceSection';

interface NodeDetailsPanelProps {
  node: GraphNodeData;
  mode: GraphMode;
  options: LabelOption[];
  isHub: boolean;
  structuralLabel?: string;
  domainLabel?: string;
  featureImportanceUnsupported?: boolean;
  backendLabel?: string;
  onChange: (newLabelValue: number) => void;
  onClose: () => void;
  leakageGold?: LeakageGoldEntry[];
  leakageGoldValue?: number;
  onLeakageChange?: (newValue: number) => void;
}

const typeIdDescriptuon = (mode: GraphMode) => {
  switch (mode) {
    case 'ast':
      return 'AST';
    case 'dataflow':
    case 'labeling':
    case 'leakage':
      return 'Dataflow';
    case 'lineage':
    case 'vamsa_lineage':
      return 'Lineage';
    case 'vamsa_wir':
      return 'Vamsa';
    default:
      return '';
  }
};

export default function NodeDetailsPanel({
  node,
  mode,
  options,
  isHub,
  structuralLabel,
  domainLabel,
  featureImportanceUnsupported = false,
  backendLabel,
  onChange,
  onClose,
  leakageGold = [],
  leakageGoldValue,
  onLeakageChange
}: NodeDetailsPanelProps) {
  const activeDropdownValue =
    node.predicted_label !== undefined ? node.predicted_label : -1;
  const detectedEntry = leakageGold.find(e => e.key === node.leakage_class);

  return (
    <PanelContainer onClose={onClose} ariaLabel="Node details">
      <PanelHeader title={node.label || 'Node Details'} onClose={onClose} />

      <PanelBody>
        {node.has_leakage && node.leakage_class && (
          <PanelSection title="⚠ Data Leakage Detected">
            <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded border border-red-300 dark:border-red-700">
              <div className="flex items-center gap-2">
                {detectedEntry && (
                  <span
                    className="w-3 h-3 rounded-sm shrink-0"
                    style={{ backgroundColor: detectedEntry.color }}
                    aria-hidden="true"
                  />
                )}
                <span className="text-sm font-bold text-red-700 dark:text-red-300">
                  {detectedEntry?.label ?? node.leakage_class}
                </span>
              </div>
              {detectedEntry && (
                <p className="mt-1 text-xs text-red-800 dark:text-red-300">
                  {detectedEntry.description}
                </p>
              )}
              <div className="mt-2 font-mono text-[10px] text-red-500 dark:text-red-400">
                {node.leakage_class}
              </div>
            </div>
          </PanelSection>
        )}

        {mode === 'leakage' && onLeakageChange && (
          <LeakageGoldSelect
            entries={leakageGold}
            value={leakageGoldValue ?? 0}
            onChange={onLeakageChange}
          />
        )}

        {mode === 'labeling' && isHub && (
          <PanelLabelSelect
            label="Node Domain Label"
            value={activeDropdownValue}
            options={options}
            onChange={onChange}
          />
        )}

        {domainLabel && (
          <PanelSection title="Domain Label">
            <PanelBadge text={domainLabel} />
          </PanelSection>
        )}

        {mode === 'labeling' && node.feature_importance && (
          <FeatureImportanceSection importance={node.feature_importance} />
        )}

        {mode === 'labeling' &&
          !node.feature_importance &&
          featureImportanceUnsupported && (
            <PanelSection title="Feature Importance">
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Feature importance isn&apos;t available for the{' '}
                <span className="font-mono">{backendLabel ?? 'selected'}</span>{' '}
                backend - use HGT or XGBoost.
              </p>
            </PanelSection>
          )}

        <PanelSection title={`${typeIdDescriptuon(mode)} Type ID`}>
          <p className="text-sm font-mono text-gray-800 dark:text-gray-300">
            {node.node_type !== undefined ? node.node_type : 'N/A'}
            {structuralLabel ? ` (${structuralLabel})` : ''}
          </p>
        </PanelSection>

        <PanelSection title="Source Code">
          <PanelCodeBlock code={node.code} />
        </PanelSection>

        {node.base_inputs && (
          <PanelSection title="Parameters / Imports">
            <div className="bg-green-50 dark:bg-green-900/20 p-2 rounded text-xs text-green-800 dark:text-green-300 border border-green-200 dark:border-green-700 font-mono overflow-x-auto whitespace-pre-wrap">
              {node.base_inputs}
            </div>
          </PanelSection>
        )}

        {(mode === 'labeling' ||
          mode === 'leakage' ||
          mode === 'lineage' ||
          mode === 'dataflow') && (
          <PanelSection title="Transformation History">
            {node.transform_history && node.transform_history.length > 0 ? (
              <div className="space-y-3">
                {node.transform_history.map((step, idx) => (
                  <div
                    key={idx}
                    className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded border border-blue-100 dark:border-blue-800 flex flex-col gap-2"
                  >
                    <div className="flex justify-between items-center border-b border-blue-200 dark:border-blue-700 pb-1">
                      <span className="text-xs font-bold text-blue-800 dark:text-blue-300 uppercase truncate pr-2">
                        {step.operation}
                      </span>
                      <span className="text-xs text-blue-600 dark:text-blue-400 font-mono whitespace-nowrap">
                        → {step.target_node}
                      </span>
                    </div>
                    {step.transform_code && (
                      <div className="text-xs text-gray-700 dark:text-gray-300 font-mono bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-600 whitespace-pre-wrap break-words">
                        {step.transform_code}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded text-xs border border-gray-200 dark:border-gray-600 border-dashed text-gray-500 dark:text-gray-400 italic">
                No linear transformations were contracted into this node.
              </div>
            )}
          </PanelSection>
        )}
      </PanelBody>
    </PanelContainer>
  );
}
