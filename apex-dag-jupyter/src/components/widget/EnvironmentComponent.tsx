import React from 'react';

interface EnvironmentData {
  imports: {
    declared: [string, string | null][];
    usage: Record<string, Record<string, number>>;
    counts: Record<string, number>;
    classes_defined: string[];
    functions_defined: string[];
  };
  complexity: Record<string, number>;
}

interface EnvironmentComponentProps {
  data: EnvironmentData | null;
}

const EnvironmentComponent: React.FC<EnvironmentComponentProps> = ({
  data
}) => {
  if (!data) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-[#fafafa] text-gray-500 italic text-sm">
        No environment data available. Execute a cell to extract telemetry.
      </div>
    );
  }

  const { imports, complexity } = data;

  const MetricCard = ({ label, value }: { label: string; value: number }) => (
    <div className="bg-white p-3 border border-gray-200 rounded shadow-sm flex flex-col justify-between">
      <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
        {label}
      </span>
      <span className="text-xl font-mono text-gray-800 mt-1">{value}</span>
    </div>
  );

  return (
    <div className="flex flex-col h-full bg-[#fafafa] overflow-y-auto p-4 space-y-6 box-border">
      <div>
        <h2 className="text-xs font-bold text-gray-800 uppercase tracking-widest mb-3 border-b border-gray-200 pb-1">
          Structural Complexity
        </h2>
        <div className="grid grid-cols-[repeat(auto-fit,minmax(160px,1fr))] gap-3">
          <MetricCard
            label="Max Nesting Depth"
            value={complexity.max_nesting_depth}
          />
          <MetricCard
            label="Branches (If/Match)"
            value={complexity.branches + complexity.match_cases}
          />
          <MetricCard label="Loops (For/While)" value={complexity.loops} />
          <MetricCard
            label="Comprehensions"
            value={
              complexity.list_comp + complexity.dict_comp + complexity.set_comp
            }
          />
          <MetricCard label="Try/Except Blocks" value={complexity.try_except} />
          <MetricCard
            label="Context Mgrs (With)"
            value={complexity.with_blocks}
          />
          <MetricCard
            label="Defined Classes"
            value={imports.classes_defined.length}
          />
          <MetricCard
            label="Defined Functions"
            value={imports.functions_defined.length}
          />
        </div>
      </div>

      <div className="flex-grow">
        <h2 className="text-xs font-bold text-gray-800 uppercase tracking-widest mb-3 border-b border-gray-200 pb-1">
          Module Dependencies
        </h2>

        {imports.declared.length === 0 ? (
          <div className="bg-white p-4 rounded border border-dashed border-gray-300 text-xs text-gray-500">
            No external libraries imported in the current context.
          </div>
        ) : (
          <div className="bg-white border border-gray-200 rounded shadow-sm overflow-x-auto flex flex-col w-full">
            <table className="min-w-full divide-y divide-gray-200 text-left text-xs">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 font-semibold text-gray-600 uppercase tracking-wider">
                    Module
                  </th>
                  <th className="px-4 py-2 font-semibold text-gray-600 uppercase tracking-wider">
                    Alias
                  </th>
                  <th className="px-4 py-2 font-semibold text-gray-600 uppercase tracking-wider">
                    Usage Count
                  </th>
                  <th className="px-4 py-2 font-semibold text-gray-600 uppercase tracking-wider w-1/2">
                    Accessed Attributes
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {imports.declared.map(([modulePath, alias], idx) => {
                  const lookupName =
                    alias || modulePath.split('.').pop() || modulePath;
                  const count = imports.counts[lookupName] || 0;
                  const usages = imports.usage[lookupName] || {};

                  const attributes = Object.keys(usages)
                    .filter(k => k !== '__direct__')
                    .sort();
                  const directCallCount = usages['__direct__'];

                  return (
                    <tr key={idx} className="hover:bg-gray-50">
                      <td className="px-4 py-2 font-mono text-blue-700">
                        {modulePath}
                      </td>
                      <td className="px-4 py-2 font-mono text-gray-500">
                        {alias || '-'}
                      </td>
                      <td className="px-4 py-2 font-mono text-gray-800">
                        {count}
                      </td>
                      <td className="px-4 py-2 text-gray-600 truncate">
                        <div className="flex flex-wrap gap-1">
                          {directCallCount > 0 && (
                            <span className="bg-gray-100 px-1.5 py-0.5 rounded border border-gray-200 text-[10px] font-mono">
                              [Direct] ({directCallCount})
                            </span>
                          )}
                          {attributes.map((attr, i) => (
                            <span
                              key={i}
                              className="bg-blue-50 px-1.5 py-0.5 rounded border border-blue-100 text-[10px] font-mono text-blue-800"
                            >
                              .{attr} ({usages[attr]})
                            </span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default EnvironmentComponent;
