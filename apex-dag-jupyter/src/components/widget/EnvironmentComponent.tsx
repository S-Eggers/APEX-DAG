import React from 'react';

export interface EnvironmentData {
  libraries: Array<{ name: string; version: string; alias?: string }>;
  functions: Array<{ name: string; callCount: number; module: string }>;
  udfs: Array<{ name: string; lines: number; complexityScore?: number }>;
}

interface Props {
  data: EnvironmentData | null;
  isLoading: boolean;
  error: string | null;
}

export const EnvironmentComponent: React.FC<Props> = ({
  data,
  isLoading,
  error
}) => {
  if (isLoading) {
    return (
      <div className="apex-env-container p-4">
        <div className="spinner">Fetching backend data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="apex-env-container p-4">
        <div className="error-state text-red-500 font-bold">Error: {error}</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="apex-env-container p-4">
        <div className="empty-state text-gray-500">
          No environment data available. Execute a cell to extract data.
        </div>
      </div>
    );
  }

  return (
    <div className="apex-env-container p-4 overflow-y-auto">
      <h2 className="text-lg font-bold mb-4">Execution Environment</h2>

      <div className="mb-6">
        <h3 className="font-semibold mb-2">
          Imported Libraries ({data.libraries.length})
        </h3>
        <div className="bg-gray-100 p-2 rounded text-sm font-mono">
          {data.libraries.length === 0
            ? 'None detected'
            : data.libraries.map((lib, i) => (
                <div key={i}>
                  {lib.name} {lib.version} {lib.alias ? `as ${lib.alias}` : ''}
                </div>
              ))}
        </div>
      </div>

      <div className="mb-6">
        <h3 className="font-semibold mb-2">
          Used Functions ({data.functions.length})
        </h3>
        <div className="bg-gray-100 p-2 rounded text-sm font-mono">
          {data.functions.length === 0
            ? 'None detected'
            : data.functions.map((fn, i) => (
                <div key={i}>
                  {fn.module}.{fn.name}() - Called {fn.callCount}x
                </div>
              ))}
        </div>
      </div>
    </div>
  );
};
