import React from 'react';
import { LabelOption } from '../../../types/GraphTypes';
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
  const getTaxonomyLabel = (typeNum: number) => `TYPE_${typeNum}`;

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

        {mode === 'labeling' && (
          <PanelLabelSelect
            label="Domain Label"
            value={data.predicted_label}
            options={options}
            onChange={onChange}
          />
        )}

        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-100">
          <PanelSection title="From">
            <PanelBadge text={edge.source().data('label')} isMono />
          </PanelSection>
          <PanelSection title="To">
            <PanelBadge text={edge.target().data('label')} isMono />
          </PanelSection>
        </div>
      </PanelBody>
    </PanelContainer>
  );
}
