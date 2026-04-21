import React from 'react';
import { LegendItemType } from '../../../types/GraphTypes';

interface GraphLegendProps {
  items: LegendItemType[];
}

export default function GraphLegend({ items }: GraphLegendProps) {
  if (items.length === 0) return null;

  return (
    <ul className="flex flex-row justify-center items-center bg-white text-gray-700 list-none p-4 m-0 flex-wrap shrink-0 shadow-inner z-10 relative">
      {items.map((item, index) => (
        <li
          key={index}
          className="mr-6 flex flex-col justify-center items-center last:mr-0 min-w-[60px]"
        >
          <div
            className={
              item.type === 'node'
                ? 'w-[50px] h-[25px] rounded m-1 inline-block border border-gray-300/50'
                : 'w-full h-0 inline-block border-t-[3px] bg-transparent mb-2 mt-2'
            }
            style={{
              backgroundColor:
                item.type === 'node' ? item.color : 'transparent',
              borderColor: item.color,
              borderStyle: item.borderStyle
            }}
          />
          <span className="text-xs font-medium tracking-wide">
            {item.label}
          </span>
        </li>
      ))}
    </ul>
  );
}
