import React from 'react';
import { LineageTuple } from '../../../types/GraphTypes';
import { useSharedLocalStorage } from '../../../hooks/useSharedState';

interface TuplesPanelProps {
  tuples: LineageTuple[];
  nodeLabels: Map<string, string>;
  onTupleClick: (subjectId: string, objectId: string) => void;
}

const TYPE_META: Record<
  string,
  { badge: string; badgeCls: string; arrow: string }
> = {
  '<D, D>': {
    badge: '<D, D>',
    badgeCls: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
    arrow: '→'
  },
  '<M, D>': {
    badge: '<M, D>',
    badgeCls:
      'bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300',
    arrow: '←'
  },
  '<D, Empty>': {
    badge: '<D, ∅>',
    badgeCls:
      'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300',
    arrow: ''
  }
};

const ORDER = ['<D, D>', '<M, D>', '<D, Empty>'];

export default function TuplesPanel({
  tuples,
  nodeLabels,
  onTupleClick
}: TuplesPanelProps) {
  const [isExpanded, setIsExpanded] = useSharedLocalStorage<boolean>(
    'systemx-tuples-expanded',
    true
  );
  const [collapsedTypes, setCollapsedTypes] = useSharedLocalStorage<string[]>(
    'systemx-tuples-collapsed-types',
    []
  );

  if (tuples.length === 0) return null;

  const grouped = new Map<string, LineageTuple[]>();
  for (const t of tuples) {
    const list = grouped.get(t.tuple_type) ?? [];
    list.push(t);
    grouped.set(t.tuple_type, list);
  }

  const toggleType = (type: string) =>
    setCollapsedTypes(prev =>
      prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type]
    );

  const resolve = (id: string) => nodeLabels.get(id) ?? id;

  return (
    <div className="absolute top-4 left-4 z-20 flex flex-col bg-white/95 dark:bg-gray-900/95 backdrop-blur shadow-lg rounded-md border border-gray-200 dark:border-gray-700 w-64 select-none">
      <div
        className="flex justify-between items-center px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-300 dark:border-gray-600 cursor-pointer rounded-t-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span className="text-xs font-bold text-gray-800 dark:text-gray-200 tracking-wider">
          TUPLES{' '}
          <span className="text-gray-500 dark:text-gray-400 font-normal ml-1">
            ({tuples.length})
          </span>
        </span>
        <span className="text-gray-600 dark:text-gray-400 text-xs">
          {isExpanded ? '▼' : '▲'}
        </span>
      </div>

      {isExpanded && (
        <div className="overflow-y-auto max-h-[50vh] custom-scrollbar">
          {ORDER.filter(type => grouped.has(type)).map(type => {
            const items = grouped.get(type)!;
            const meta = TYPE_META[type] ?? TYPE_META['<D, Empty>'];
            const isCollapsed = collapsedTypes.includes(type);

            return (
              <div
                key={type}
                className="border-b border-gray-100 dark:border-gray-700 last:border-0"
              >
                <div
                  className="flex justify-between items-center px-3 py-1.5 bg-gray-50 dark:bg-gray-800 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                  onClick={() => toggleType(type)}
                >
                  <span
                    className={`text-[10px] font-bold px-1.5 py-0.5 rounded font-mono ${meta.badgeCls}`}
                  >
                    {meta.badge}
                  </span>
                  <span className="text-gray-400 dark:text-gray-500 text-[10px] ml-auto mr-2">
                    {items.length}
                  </span>
                  <span className="text-gray-400 dark:text-gray-500 text-[10px]">
                    {isCollapsed ? '+' : '-'}
                  </span>
                </div>

                {!isCollapsed && (
                  <ul className="list-none m-0 px-3 py-1.5 space-y-1 bg-white dark:bg-gray-900">
                    {items.map((tuple, idx) => (
                      <li
                        key={idx}
                        className="flex items-center gap-1.5 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 px-1 py-1 -mx-1 rounded transition-colors"
                        onClick={() =>
                          onTupleClick(tuple.subject_id, tuple.object_id)
                        }
                        title={`${tuple.subject_id} ${meta.arrow} ${tuple.object_id}`}
                      >
                        <span className="text-[11px] font-mono text-gray-700 dark:text-gray-300 truncate flex-1 min-w-0">
                          {resolve(tuple.subject_id)}
                        </span>
                        {meta.arrow && (
                          <>
                            <span className="text-[10px] text-gray-400 dark:text-gray-500 shrink-0">
                              {meta.arrow}
                            </span>
                            <span className="text-[11px] font-mono text-gray-700 dark:text-gray-300 truncate flex-1 min-w-0">
                              {resolve(tuple.object_id)}
                            </span>
                          </>
                        )}
                        {!meta.arrow && (
                          <span className="text-[10px] text-gray-400 dark:text-gray-500 shrink-0 italic">
                            no consumer
                          </span>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
