import React from 'react';
import { LegendItemType } from '../../../types/GraphTypes';
import { useSharedLocalStorage } from '../../../hooks/useSharedState';
import { adjustForDarkMode } from '../../../utils/colorTheme';

interface GraphLegendProps {
  groupedItems: Record<string, LegendItemType[]>;
  onItemClick?: (item: LegendItemType) => void;
  isDark?: boolean;
}

export default function GraphLegend({
  groupedItems,
  onItemClick,
  isDark = false
}: GraphLegendProps) {
  const [isMainExpanded, setIsMainExpanded] = useSharedLocalStorage<boolean>(
    'systemx-legend-main-expanded',
    true
  );

  const [collapsedCategories, setCollapsedCategories] = useSharedLocalStorage<
    string[]
  >('systemx-legend-collapsed-cats', []);

  const toggleCategory = (category: string) => {
    setCollapsedCategories(prev =>
      prev.includes(category)
        ? prev.filter(c => c !== category)
        : [...prev, category]
    );
  };

  const onKeyActivate =
    (fn: () => void) => (e: React.KeyboardEvent<HTMLElement>) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fn();
      }
    };

  const categories = Object.keys(groupedItems).sort((a, b) => {
    if (a === 'Hub Nodes') return -1;
    if (b === 'Hub Nodes') return 1;
    return a.localeCompare(b);
  });

  if (categories.length === 0) return null;

  const totalElements = Object.values(groupedItems)
    .flat()
    .reduce((sum, item) => sum + (item.count || 0), 0);

  return (
    <div className="absolute bottom-6 left-6 z-20 flex flex-col bg-white/95 dark:bg-gray-900/95 backdrop-blur shadow-lg rounded-md border border-gray-200 dark:border-gray-700 w-64 select-none">
      <div
        className="flex justify-between items-center px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-300 dark:border-gray-600 cursor-pointer rounded-t-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
        onClick={() => setIsMainExpanded(!isMainExpanded)}
        onKeyDown={onKeyActivate(() => setIsMainExpanded(!isMainExpanded))}
        role="button"
        tabIndex={0}
        aria-expanded={isMainExpanded}
        aria-label="Toggle legend"
      >
        <span className="text-xs font-bold text-gray-800 dark:text-gray-200 tracking-wider">
          LEGEND{' '}
          <span className="text-gray-500 dark:text-gray-400 font-normal ml-1">
            ({totalElements})
          </span>
        </span>
        <span className="text-gray-600 dark:text-gray-400 text-xs">
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
                className="border-b border-gray-100 dark:border-gray-700 last:border-0"
              >
                <div
                  className="flex justify-between items-center px-4 py-1.5 bg-gray-50 dark:bg-gray-800 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                  onClick={() => toggleCategory(category)}
                  onKeyDown={onKeyActivate(() => toggleCategory(category))}
                  role="button"
                  tabIndex={0}
                  aria-expanded={!isCollapsed}
                  aria-label={`Toggle ${category} category`}
                >
                  <span className="text-[11px] font-semibold text-gray-600 dark:text-gray-400">
                    {category}
                  </span>
                  <span className="text-gray-400 text-[10px]">
                    {isCollapsed ? '+' : '-'}
                  </span>
                </div>

                {!isCollapsed && (
                  <ul className="list-none m-0 px-4 py-2 space-y-2 bg-white dark:bg-gray-900">
                    {items.map((item, index) => {
                      const isHexagon =
                        item.type === 'node' && category === 'Hub Nodes';
                      const swatchColor =
                        item.type === 'node' && isDark
                          ? adjustForDarkMode(item.color)
                          : item.color;

                      return (
                        <li
                          key={index}
                          className="flex items-center cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 p-1 -mx-1 rounded transition-colors"
                          onClick={() => onItemClick?.(item)}
                          onKeyDown={
                            onItemClick
                              ? onKeyActivate(() => onItemClick(item))
                              : undefined
                          }
                          role={onItemClick ? 'button' : undefined}
                          tabIndex={onItemClick ? 0 : undefined}
                          aria-label={
                            onItemClick
                              ? `Highlight ${item.label}${
                                  item.count !== undefined
                                    ? `, ${item.count} elements`
                                    : ''
                                }`
                              : undefined
                          }
                        >
                          {' '}
                          {isHexagon ? (
                            <div
                              className="w-4 h-4 mr-3 shrink-0 drop-shadow-sm"
                              style={{
                                backgroundColor: swatchColor,
                                clipPath:
                                  'polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%)'
                              }}
                            />
                          ) : (
                            <div
                              className={
                                item.type === 'node'
                                  ? 'w-4 h-3 rounded-sm border border-gray-300 mr-3 shrink-0 shadow-sm'
                                  : 'w-4 h-0 border-t-2 mr-3 shrink-0'
                              }
                              style={{
                                backgroundColor:
                                  item.type === 'node'
                                    ? swatchColor
                                    : 'transparent',
                                borderColor: swatchColor,
                                borderStyle: item.borderStyle
                              }}
                            />
                          )}
                          <span
                            className="text-[11px] font-medium text-gray-700 dark:text-gray-300 truncate flex-1"
                            title={item.label}
                          >
                            {item.label}
                          </span>
                          {item.count !== undefined && (
                            <span className="ml-2 text-[10px] text-gray-500 dark:text-gray-400 font-mono bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded-full border border-gray-200 dark:border-gray-600">
                              {item.count}
                            </span>
                          )}
                        </li>
                      );
                    })}
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
