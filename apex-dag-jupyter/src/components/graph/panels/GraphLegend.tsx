import React from 'react';
import { LegendItemType } from '../../../types/GraphTypes';
import { useSharedLocalStorage } from '../../../hooks/useSharedState';

interface GraphLegendProps {
  groupedItems: Record<string, LegendItemType[]>;
}

export default function GraphLegend({ groupedItems }: GraphLegendProps) {
  const [isMainExpanded, setIsMainExpanded] = useSharedLocalStorage<boolean>(
    'apex-dag-legend-main-expanded',
    true
  );

  const [collapsedCategories, setCollapsedCategories] = useSharedLocalStorage<
    string[]
  >('apex-dag-legend-collapsed-cats', []);

  const toggleCategory = (category: string) => {
    setCollapsedCategories(prev =>
      prev.includes(category)
        ? prev.filter(c => c !== category)
        : [...prev, category]
    );
  };

  const categories = Object.keys(groupedItems);
  if (categories.length === 0) return null;

  const totalItems = Object.values(groupedItems).reduce(
    (acc, arr) => acc + arr.length,
    0
  );

  return (
    <div className="absolute bottom-6 left-6 z-20 flex flex-col bg-white/95 backdrop-blur shadow-lg rounded-md border border-gray-200 w-64 select-none">
      <div
        className="flex justify-between items-center px-4 py-2 bg-gray-100 border-b border-gray-300 cursor-pointer rounded-t-md hover:bg-gray-200 transition-colors"
        onClick={() => setIsMainExpanded(!isMainExpanded)}
      >
        <span className="text-xs font-bold text-gray-800 uppercase tracking-wider">
          Legend ({totalItems})
        </span>
        <span className="text-gray-600 text-xs">
          {isMainExpanded ? '▼' : '▲'}
        </span>
      </div>

      {isMainExpanded && (
        <div className="overflow-y-auto max-h-[50vh] custom-scrollbar">
          {categories.map(category => {
            const items = groupedItems[category];
            const isCollapsed = collapsedCategories.includes(category);

            return (
              <div
                key={category}
                className="border-b border-gray-100 last:border-0"
              >
                <div
                  className="flex justify-between items-center px-4 py-1.5 bg-gray-50 cursor-pointer hover:bg-gray-100 transition-colors"
                  onClick={() => toggleCategory(category)}
                >
                  <span className="text-[11px] font-semibold text-gray-600">
                    {category}
                  </span>
                  <span className="text-gray-400 text-[10px]">
                    {isCollapsed ? '+' : '-'}
                  </span>
                </div>

                {!isCollapsed && (
                  <ul className="list-none m-0 px-4 py-2 space-y-2 bg-white">
                    {items.map((item, index) => (
                      <li key={index} className="flex items-center">
                        <div
                          className={
                            item.type === 'node'
                              ? 'w-4 h-3 rounded-sm border border-gray-300 mr-3 shrink-0 shadow-sm'
                              : 'w-4 h-0 border-t-2 mr-3 shrink-0'
                          }
                          style={{
                            backgroundColor:
                              item.type === 'node' ? item.color : 'transparent',
                            borderColor: item.color,
                            borderStyle: item.borderStyle
                          }}
                        />
                        <span
                          className="text-[11px] font-medium text-gray-700 truncate"
                          title={item.label}
                        >
                          {item.label}
                        </span>
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
