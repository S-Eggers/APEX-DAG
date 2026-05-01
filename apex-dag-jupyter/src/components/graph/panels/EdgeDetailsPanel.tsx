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
  const data = edge.data();

  const getTaxonomyLabel = (typeNum: number | undefined) => {
    if (typeNum === undefined) return 'UNKNOWN_TYPE';
    return `TYPE_${typeNum}`;
  };

  // Cytoscape's source() and target() return CyElements. Call .data() on them.
  const sourceData = edge.source?.()?.data();
  const targetData = edge.target?.()?.data();

  return (
    <PanelContainer>
      <PanelHeader title="Edge Details" onClose={onClose} />

      <PanelBody>
        <div className="grid grid-cols-2 gap-4">
          <PanelSection title="Structural Type">
            <PanelBadge text={getTaxonomyLabel(data.edge_type)} />
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

        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-100">
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
