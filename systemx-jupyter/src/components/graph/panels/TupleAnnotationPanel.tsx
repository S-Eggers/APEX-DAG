import React from 'react';
import { LineageTuple } from '../../../types/GraphTypes';
import { useSharedLocalStorage } from '../../../hooks/useSharedState';

type TupleType = LineageTuple['tuple_type'];

interface TupleAnnotationPanelProps {
  tuples: LineageTuple[];
  nodeLabels: Map<string, string>;
  subjectId?: string;
  objectId?: string;
  onAddTuple: (tupleType: TupleType) => void;
  onDeleteTuple: (index: number) => void;
  onClearPicks: () => void;
  onTupleClick: (subjectId: string, objectId: string) => void;
}

const TYPE_META: Record<TupleType, { badge: string; badgeCls: string; arrow: string }> =
  {
    '<D, D>': {
      badge: '<D, D>',
      badgeCls:
        'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
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
      badgeCls: 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300',
      arrow: ''
    }
  };

const ORDER: TupleType[] = ['<D, D>', '<M, D>', '<D, Empty>'];

export default function TupleAnnotationPanel({
  tuples,
  nodeLabels,
  subjectId,
  objectId,
  onAddTuple,
  onDeleteTuple,
  onClearPicks,
  onTupleClick
}: TupleAnnotationPanelProps) {
  const [tupleType, setTupleType] = useSharedLocalStorage<TupleType>(
    'systemx-tuple-annot-type',
    '<D, D>'
  );
  const [collapsedTypes, setCollapsedTypes] = useSharedLocalStorage<string[]>(
    'systemx-tuple-annot-collapsed-types',
    []
  );

  const resolve = (id: string) => nodeLabels.get(id) ?? id;

  const needsObject = tupleType !== '<D, Empty>';
  const canAdd = !!subjectId && (!needsObject || !!objectId);

  const toggleType = (type: string) =>
    setCollapsedTypes(prev =>
      prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type]
    );

  const grouped = new Map<TupleType, { tuple: LineageTuple; index: number }[]>();
  tuples.forEach((tuple, index) => {
    const list = grouped.get(tuple.tuple_type) ?? [];
    list.push({ tuple, index });
    grouped.set(tuple.tuple_type, list);
  });

  const meta = TYPE_META[tupleType];

  return (
    <div className="absolute top-4 right-4 z-20 flex flex-col bg-white/95 dark:bg-gray-900/95 backdrop-blur shadow-lg rounded-md border border-gray-200 dark:border-gray-700 w-72 select-none">
      <div className="flex justify-between items-center px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-300 dark:border-gray-600 rounded-t-md">
        <span className="text-xs font-bold text-gray-800 dark:text-gray-200 tracking-wider">
          TUPLE ANNOTATION{' '}
          <span className="text-gray-500 dark:text-gray-400 font-normal ml-1">
            ({tuples.length})
          </span>
        </span>
      </div>

      {/* Builder: pick buffer + type + Add */}
      <div className="px-3 py-2 border-b border-gray-200 dark:border-gray-700 space-y-2">
        <p className="text-[10px] text-gray-500 dark:text-gray-400 leading-snug">
          Tap the <span className="text-green-600 font-semibold">subject</span>{' '}
          node, then the{' '}
          <span className="text-amber-600 font-semibold">object</span> node on the
          graph, choose a type, and add the tuple.
        </p>

        <div className="flex items-center gap-1.5 text-[11px] font-mono">
          <span
            className={`flex-1 min-w-0 truncate px-1.5 py-1 rounded border ${
              subjectId
                ? 'bg-green-50 dark:bg-green-900/30 border-green-300 dark:border-green-700 text-green-800 dark:text-green-300'
                : 'bg-gray-50 dark:bg-gray-800 border-dashed border-gray-300 dark:border-gray-600 text-gray-400 italic'
            }`}
            title={subjectId ? resolve(subjectId) : undefined}
          >
            {subjectId ? resolve(subjectId) : 'subject'}
          </span>
          <span className="text-gray-400 shrink-0">→</span>
          {needsObject ? (
            <span
              className={`flex-1 min-w-0 truncate px-1.5 py-1 rounded border ${
                objectId
                  ? 'bg-amber-50 dark:bg-amber-900/30 border-amber-300 dark:border-amber-700 text-amber-800 dark:text-amber-300'
                  : 'bg-gray-50 dark:bg-gray-800 border-dashed border-gray-300 dark:border-gray-600 text-gray-400 italic'
              }`}
              title={objectId ? resolve(objectId) : undefined}
            >
              {objectId ? resolve(objectId) : 'object'}
            </span>
          ) : (
            <span className="flex-1 min-w-0 truncate px-1.5 py-1 rounded border border-dashed border-gray-300 dark:border-gray-600 text-gray-400 italic">
              ∅ no consumer
            </span>
          )}
        </div>

        <div className="flex items-center gap-1.5">
          <select
            value={tupleType}
            onChange={e => setTupleType(e.target.value as TupleType)}
            className="text-[11px] font-mono rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200 px-1.5 py-1"
            aria-label="Tuple type"
          >
            {ORDER.map(t => (
              <option key={t} value={t}>
                {TYPE_META[t].badge}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={() => canAdd && onAddTuple(tupleType)}
            disabled={!canAdd}
            className={`flex-1 text-xs font-medium px-2 py-1 rounded transition-colors !text-white ${
              canAdd
                ? 'bg-indigo-600 hover:bg-indigo-700 cursor-pointer'
                : 'bg-gray-300 dark:bg-gray-700 !text-gray-400 cursor-not-allowed'
            }`}
          >
            + Add {meta.badge}
          </button>

          <button
            type="button"
            onClick={onClearPicks}
            disabled={!subjectId && !objectId}
            title="Clear the current subject/object picks"
            className="text-xs px-2 py-1 rounded border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Collected tuples */}
      {tuples.length === 0 ? (
        <div className="px-3 py-4 text-[11px] text-gray-400 dark:text-gray-500 italic text-center">
          No tuples yet.
        </div>
      ) : (
        <div className="overflow-y-auto max-h-[45vh] custom-scrollbar">
          {ORDER.filter(type => grouped.has(type)).map(type => {
            const items = grouped.get(type)!;
            const m = TYPE_META[type];
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
                    className={`text-[10px] font-bold px-1.5 py-0.5 rounded font-mono ${m.badgeCls}`}
                  >
                    {m.badge}
                  </span>
                  <span className="text-gray-400 dark:text-gray-500 text-[10px] ml-auto mr-2">
                    {items.length}
                  </span>
                  <span className="text-gray-400 dark:text-gray-500 text-[10px]">
                    {isCollapsed ? '+' : '-'}
                  </span>
                </div>

                {!isCollapsed && (
                  <ul className="list-none m-0 px-2 py-1.5 space-y-1 bg-white dark:bg-gray-900">
                    {items.map(({ tuple, index }) => (
                      <li
                        key={index}
                        className="flex items-center gap-1.5 hover:bg-gray-50 dark:hover:bg-gray-800 px-1 py-1 rounded transition-colors group"
                      >
                        <button
                          type="button"
                          className="flex items-center gap-1.5 flex-1 min-w-0 text-left cursor-pointer"
                          onClick={() =>
                            onTupleClick(tuple.subject_id, tuple.object_id)
                          }
                          title={`${tuple.subject_id} ${m.arrow} ${tuple.object_id}`}
                        >
                          <span className="text-[11px] font-mono text-gray-700 dark:text-gray-300 truncate flex-1 min-w-0">
                            {resolve(tuple.subject_id)}
                          </span>
                          {m.arrow ? (
                            <>
                              <span className="text-[10px] text-gray-400 shrink-0">
                                {m.arrow}
                              </span>
                              <span className="text-[11px] font-mono text-gray-700 dark:text-gray-300 truncate flex-1 min-w-0">
                                {resolve(tuple.object_id)}
                              </span>
                            </>
                          ) : (
                            <span className="text-[10px] text-gray-400 shrink-0 italic">
                              no consumer
                            </span>
                          )}
                        </button>
                        <button
                          type="button"
                          onClick={() => onDeleteTuple(index)}
                          title="Delete tuple"
                          aria-label="Delete tuple"
                          className="shrink-0 text-gray-400 hover:text-red-600 dark:hover:text-red-400 text-xs px-1 opacity-60 group-hover:opacity-100 transition-opacity"
                        >
                          ✕
                        </button>
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
