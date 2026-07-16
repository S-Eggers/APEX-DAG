import React from 'react';
import { useDarkMode } from '../../hooks/useDarkMode';

interface EnvironmentData {
  imports: {
    declared: [string, string | null][];
    usage: Record<string, Record<string, number>>;
    counts: Record<string, number>;
    classes_defined: string[];
    functions_defined: string[];
    versions: Record<string, string | null>;
  };
  complexity: Record<string, number>;
  runtime: {
    python_version: string;
    lines_of_code: number;
    code_cells: number;
  };
}

interface EnvironmentComponentProps {
  data: EnvironmentData | null;
}

const MetricCard = ({
  label,
  value,
  accent
}: {
  label: string;
  value: React.ReactNode;
  accent?: string;
}) => (
  <div className="bg-white dark:bg-[#21262d] p-3 border border-gray-200 dark:border-gray-700 rounded shadow-sm flex flex-col justify-between">
    <span className="text-[10px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-wider">
      {label}
    </span>
    <span
      className={`text-xl font-mono mt-1 ${accent ?? 'text-gray-800 dark:text-gray-200'}`}
    >
      {value}
    </span>
  </div>
);

const SectionHeader = ({ children }: { children: React.ReactNode }) => (
  <h2 className="text-xs font-bold text-gray-800 dark:text-gray-200 uppercase tracking-widest mb-3 border-b border-gray-200 dark:border-gray-700 pb-1">
    {children}
  </h2>
);

const VersionBadge = ({ version }: { version: string | null | undefined }) => {
  if (version === 'stdlib') {
    return (
      <span className="bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 px-1.5 py-0.5 rounded border border-gray-200 dark:border-gray-600 text-[10px] font-mono">
        stdlib
      </span>
    );
  }
  if (!version) {
    return (
      <span
        className="text-gray-300 dark:text-gray-600 text-[11px] font-mono"
        title="Version could not be resolved in the server environment"
      >
        -
      </span>
    );
  }
  return (
    <span className="bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300 px-1.5 py-0.5 rounded border border-green-100 dark:border-green-800 text-[10px] font-mono">
      {version}
    </span>
  );
};

const EnvironmentComponent: React.FC<EnvironmentComponentProps> = ({
  data
}) => {
  const isDark = useDarkMode();

  if (!data) {
    return (
      <div
        className={`${isDark ? 'dark ' : ''}flex h-full w-full items-center justify-center bg-[#eef2f8] dark:bg-[#0d1117] text-gray-500 dark:text-gray-400 italic text-sm`}
      >
        No environment data available. Execute a cell to extract telemetry.
      </div>
    );
  }

  const { imports, complexity, runtime } = data;

  const lookupCount = (modulePath: string, alias: string | null): number => {
    const lookupName = alias || modulePath.split('.').pop() || modulePath;
    return imports.counts[lookupName] || 0;
  };

  const unusedCount = imports.declared.filter(
    ([m, a]) => lookupCount(m, a) === 0
  ).length;

  const cyclomatic =
    1 +
    (complexity.branches || 0) +
    (complexity.loops || 0) +
    (complexity.match_cases || 0) +
    (complexity.try_except || 0) +
    (complexity.list_comp || 0) +
    (complexity.dict_comp || 0) +
    (complexity.set_comp || 0) +
    (complexity.gen_expr || 0);

  return (
    <div
      className={`${isDark ? 'dark ' : ''}flex flex-col h-full bg-[#eef2f8] dark:bg-[#0d1117] overflow-y-auto p-4 space-y-6 box-border`}
    >
      <div>
        <SectionHeader>Runtime</SectionHeader>
        <div className="grid grid-cols-[repeat(auto-fit,minmax(160px,1fr))] gap-3">
          <MetricCard label="Python" value={runtime.python_version || '-'} />
          <MetricCard label="Lines of Code" value={runtime.lines_of_code} />
          <MetricCard label="Code Cells" value={runtime.code_cells} />
          <MetricCard
            label="Unused Imports"
            value={unusedCount}
            accent={
              unusedCount > 0
                ? 'text-amber-600 dark:text-amber-400'
                : 'text-gray-800 dark:text-gray-200'
            }
          />
        </div>
      </div>

      <div>
        <SectionHeader>Structural Complexity</SectionHeader>
        <div className="grid grid-cols-[repeat(auto-fit,minmax(160px,1fr))] gap-3">
          <MetricCard
            label="Max Nesting Depth"
            value={complexity.max_nesting_depth}
          />
          <MetricCard label="Cyclomatic (est.)" value={cyclomatic} />
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
          <MetricCard label="Generator Exprs" value={complexity.gen_expr} />
          <MetricCard label="Try/Except Blocks" value={complexity.try_except} />
          <MetricCard
            label="Context Mgrs (With)"
            value={complexity.with_blocks}
          />
          <MetricCard
            label="Async Functions"
            value={complexity.async_functions ?? 0}
          />
          <MetricCard label="Awaits" value={complexity.awaits ?? 0} />
        </div>
      </div>

      {(imports.classes_defined.length > 0 ||
        imports.functions_defined.length > 0) && (
        <div>
          <SectionHeader>Definitions</SectionHeader>
          <div className="space-y-3">
            {imports.classes_defined.length > 0 && (
              <div>
                <div className="text-[10px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-wider mb-1.5">
                  Classes ({imports.classes_defined.length})
                </div>
                <div className="flex flex-wrap gap-1">
                  {imports.classes_defined.map((c, i) => (
                    <span
                      key={i}
                      className="bg-purple-50 dark:bg-purple-900/30 text-purple-800 dark:text-purple-300 px-1.5 py-0.5 rounded border border-purple-100 dark:border-purple-800 text-[11px] font-mono"
                    >
                      {c}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {imports.functions_defined.length > 0 && (
              <div>
                <div className="text-[10px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-wider mb-1.5">
                  Functions ({imports.functions_defined.length})
                </div>
                <div className="flex flex-wrap gap-1">
                  {imports.functions_defined.map((f, i) => (
                    <span
                      key={i}
                      className="bg-blue-50 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 px-1.5 py-0.5 rounded border border-blue-100 dark:border-blue-800 text-[11px] font-mono"
                    >
                      {f}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="flex-grow">
        <SectionHeader>Module Dependencies</SectionHeader>

        {imports.declared.length === 0 ? (
          <div className="bg-white dark:bg-[#21262d] p-4 rounded border border-dashed border-gray-300 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
            No external libraries imported in the current context.
          </div>
        ) : (
          <div className="bg-white dark:bg-[#21262d] border border-gray-200 dark:border-gray-700 rounded shadow-sm overflow-x-auto flex flex-col w-full">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700 text-left text-xs">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-4 py-2 font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wider">
                    Module
                  </th>
                  <th className="px-4 py-2 font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wider">
                    Version
                  </th>
                  <th className="px-4 py-2 font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wider">
                    Alias
                  </th>
                  <th className="px-4 py-2 font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wider">
                    Usage Count
                  </th>
                  <th className="px-4 py-2 font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wider w-1/2">
                    Accessed Attributes
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {imports.declared.map(([modulePath, alias], idx) => {
                  const lookupName =
                    alias || modulePath.split('.').pop() || modulePath;
                  const count = imports.counts[lookupName] || 0;
                  const usages = imports.usage[lookupName] || {};
                  const version = imports.versions[modulePath.split('.')[0]];
                  const unused = count === 0;

                  const attributes = Object.keys(usages)
                    .filter(k => k !== '__direct__')
                    .sort();
                  const directCallCount = usages['__direct__'];

                  return (
                    <tr
                      key={idx}
                      className={
                        unused
                          ? 'bg-amber-50/50 dark:bg-amber-900/20'
                          : 'hover:bg-gray-50 dark:hover:bg-gray-800'
                      }
                    >
                      <td className="px-4 py-2 font-mono text-blue-700 dark:text-blue-400">
                        {modulePath}
                      </td>
                      <td className="px-4 py-2">
                        <VersionBadge version={version} />
                      </td>
                      <td className="px-4 py-2 font-mono text-gray-500 dark:text-gray-400">
                        {alias || '-'}
                      </td>
                      <td className="px-4 py-2 font-mono text-gray-800 dark:text-gray-200 whitespace-nowrap">
                        {count}
                        {unused && (
                          <span className="ml-2 bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 px-1.5 py-0.5 rounded border border-amber-200 dark:border-amber-700 text-[10px] font-sans uppercase tracking-wide">
                            unused
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-2 text-gray-600 dark:text-gray-400 truncate">
                        <div className="flex flex-wrap gap-1">
                          {directCallCount > 0 && (
                            <span className="bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded border border-gray-200 dark:border-gray-600 text-[10px] font-mono text-gray-600 dark:text-gray-300">
                              [Direct] ({directCallCount})
                            </span>
                          )}
                          {attributes.map((attr, i) => (
                            <span
                              key={i}
                              className="bg-blue-50 dark:bg-blue-900/30 px-1.5 py-0.5 rounded border border-blue-100 dark:border-blue-800 text-[10px] font-mono text-blue-800 dark:text-blue-300"
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
