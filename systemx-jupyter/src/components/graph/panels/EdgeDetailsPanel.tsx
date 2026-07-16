import React from 'react';
import {
  GraphMode,
  LabelOption,
  CyElement,
  GraphEdgeData
} from '../../../types/GraphTypes';
import {
  PanelContainer,
  PanelHeader,
  PanelBody,
  PanelSection,
  PanelCodeBlock,
  PanelLabelSelect,
  PanelBadge
} from './PanelComponents';

interface EdgeDetailsPanelProps {
  edge: CyElement<GraphEdgeData>;
  mode: GraphMode;
  options: LabelOption[];
  operationLabel?: string;
  structuralLabel?: string;
  onChange: (newLabelValue: number) => void;
  onClose: () => void;
}

export default function EdgeDetailsPanel({
  edge,
  mode,
  options,
  operationLabel,
  structuralLabel,
  onChange,
  onClose
}: EdgeDetailsPanelProps) {
  const data = edge.data();

  const typeLabel = operationLabel ?? structuralLabel ?? 'Unlabelled';

  const sourceData = edge.source?.()?.data();
  const targetData = edge.target?.()?.data();

  return (
    <PanelContainer onClose={onClose} ariaLabel="Edge details">
      <PanelHeader title="Edge Details" onClose={onClose} />

      <PanelBody>
        <div className="grid grid-cols-2 gap-4">
          <PanelSection title="Operation Type">
            <PanelBadge text={typeLabel} />
          </PanelSection>

          <PanelSection title="Operation">
            <PanelBadge text={data.label || 'edge'} isMono />
          </PanelSection>
        </div>

        <PanelSection title="Source Code">
          <PanelCodeBlock code={data.raw_code || data.label} />
        </PanelSection>

        {mode === 'labeling' && data.predicted_label !== undefined && (
          <PanelLabelSelect
            label="Domain Label"
            value={data.predicted_label}
            options={options}
            onChange={onChange}
          />
        )}

        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-100 dark:border-gray-700">
          <PanelSection title="From">
            <PanelBadge text={sourceData?.label || 'Unknown'} isMono />
          </PanelSection>
          <PanelSection title="To">
            <PanelBadge text={targetData?.label || 'Unknown'} isMono />
          </PanelSection>
        </div>
      </PanelBody>
    </PanelContainer>
  );
}
