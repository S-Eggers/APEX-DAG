import React from 'react';
import Graph from '../graph/Graph';
import { GraphComponentProps } from '../../types/GraphTypes';
import { useGraphTaxonomy } from '../../hooks/useGraphTaxonomy';
import { useDarkMode } from '../../hooks/useDarkMode';

const GRAPH_SURFACE_DARK = '#0d1117';
const GRAPH_SURFACE_LIGHT = '#eef2f8';

const GraphComponent: React.FC<GraphComponentProps> = ({
  graphData,
  mode,
  resetTrigger,
  notebookName,
  notebookCode,
  labelingConfig,
  nnBackend,
  featurePreset,
  modelVariant,
  explainFeatureImportance,
  backendError,
  onLocateCell,
  onNextNotebook,
  onRelabel,
  onFocusNode,
  onVariantTrained,
  tuples
}) => {
  const taxonomy = useGraphTaxonomy(mode);
  const isDark = useDarkMode();
  const surface = isDark ? GRAPH_SURFACE_DARK : GRAPH_SURFACE_LIGHT;

  if (!taxonomy.isLoaded) {
    return (
      <div
        className="flex items-center justify-center h-full w-full font-mono text-sm"
        style={{
          backgroundColor: surface,
          color: 'var(--jp-ui-font-color2)'
        }}
      >
        {taxonomy.error ? `⚠ ${taxonomy.error}` : 'Loading...'}
      </div>
    );
  }

  return (
    <div
      className="flex flex-col h-full max-h-full w-full max-w-full"
      style={{ backgroundColor: surface }}
    >
      <Graph
        graphData={graphData}
        mode={mode}
        resetTrigger={resetTrigger}
        taxonomy={taxonomy}
        notebookName={notebookName}
        notebookCode={notebookCode}
        labelingConfig={labelingConfig}
        nnBackend={nnBackend}
        featurePreset={featurePreset}
        modelVariant={modelVariant}
        explainFeatureImportance={explainFeatureImportance}
        backendError={backendError}
        onLocateCell={onLocateCell}
        onNextNotebook={onNextNotebook}
        onRelabel={onRelabel}
        onFocusNode={onFocusNode}
        onVariantTrained={onVariantTrained}
        tuples={tuples}
      />
    </div>
  );
};

export default GraphComponent;
