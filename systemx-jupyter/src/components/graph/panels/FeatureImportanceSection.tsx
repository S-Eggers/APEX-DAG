import React, { useState } from 'react';
import {
  FeatureImportance,
  FeatureGroupImportance
} from '../../../types/GraphTypes';
import { PanelSection } from './PanelComponents';

const pct = (score: number): string => `${(score * 100).toFixed(1)}%`;

const fmtVal = (v: number): string =>
  Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/\.?0+$/, '');

function Bar({
  label,
  score,
  value,
  title,
  emphasis = false
}: {
  label: string;
  score: number;
  value?: number;
  title?: string;
  emphasis?: boolean;
}) {
  const width = `${Math.max(score * 100, 1.5)}%`;
  return (
    <div className="flex flex-col gap-0.5" title={title}>
      <div className="flex justify-between items-baseline gap-2">
        <span className="flex items-baseline gap-1.5 min-w-0">
          <span
            className={`truncate ${
              emphasis
                ? 'text-xs font-semibold text-gray-800 dark:text-gray-200'
                : 'text-[11px] font-mono text-gray-600 dark:text-gray-400'
            }`}
          >
            {label}
          </span>
          {value !== undefined && (
            <span className="text-[10px] font-mono text-indigo-600 dark:text-indigo-400 shrink-0">
              = {fmtVal(value)}
            </span>
          )}
        </span>
        <span className="text-[11px] tabular-nums text-gray-500 dark:text-gray-400 shrink-0">
          {pct(score)}
        </span>
      </div>
      <div
        className="h-1.5 w-full rounded bg-gray-100 dark:bg-gray-700 overflow-hidden"
        role="progressbar"
        aria-label={`${label} importance`}
        aria-valuenow={Math.round(score * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className={`h-full rounded ${
            emphasis
              ? 'bg-indigo-500 dark:bg-indigo-400'
              : 'bg-indigo-300 dark:bg-indigo-600'
          }`}
          style={{ width }}
        />
      </div>
    </div>
  );
}

function GroupRow({ group }: { group: FeatureGroupImportance }) {
  const [open, setOpen] = useState(false);
  const drill = group.dims ?? group.scalars ?? [];
  const expandable = drill.length > 0;

  return (
    <div>
      <div
        className={`flex flex-col gap-0.5 ${
          expandable ? 'cursor-pointer' : ''
        }`}
        onClick={expandable ? () => setOpen(v => !v) : undefined}
        onKeyDown={
          expandable
            ? e => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setOpen(v => !v);
                }
              }
            : undefined
        }
        role={expandable ? 'button' : undefined}
        tabIndex={expandable ? 0 : undefined}
        aria-expanded={expandable ? open : undefined}
        aria-label={
          expandable ? `${group.name}, ${pct(group.score)}` : undefined
        }
      >
        <div className="flex justify-between items-baseline gap-2">
          <span className="truncate text-xs font-semibold text-gray-800 dark:text-gray-200">
            {expandable && (
              <span className="inline-block text-gray-400 mr-1 select-none">
                {open ? '▾' : '▸'}
              </span>
            )}
            {group.name}
          </span>
          <span className="text-[11px] tabular-nums text-gray-500 dark:text-gray-400 shrink-0">
            {pct(group.score)}
          </span>
        </div>
        <div
          className="h-2 w-full rounded bg-gray-100 dark:bg-gray-700 overflow-hidden"
          role="progressbar"
          aria-label={`${group.name} importance`}
          aria-valuenow={Math.round(group.score * 100)}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div
            className="h-full rounded bg-indigo-500 dark:bg-indigo-400"
            style={{ width: `${Math.max(group.score * 100, 1.5)}%` }}
            title={group.description}
          />
        </div>
      </div>

      {open && expandable && (
        <div className="mt-2 ml-4 flex flex-col gap-2 border-l border-gray-200 dark:border-gray-700 pl-3">
          {group.dims?.map(d => (
            <Bar key={d.index} label={d.name} score={d.score} value={d.value} />
          ))}
          {group.scalars?.map(s => (
            <Bar
              key={s.name}
              label={s.name}
              score={s.score}
              value={s.value}
              title={s.description}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function FeatureImportanceSection({
  importance
}: {
  importance: FeatureImportance;
}) {
  if (!importance?.groups?.length) {
    return null;
  }
  const groups = [...importance.groups].sort((a, b) => b.score - a.score);

  return (
    <PanelSection title="Feature Importance - why this label?">
      {importance.model && (
        <p className="text-[10px] text-gray-400 dark:text-gray-500 mb-2 -mt-1">
          {importance.model}
        </p>
      )}
      <div className="flex flex-col gap-3">
        {groups.map(g => (
          <GroupRow key={g.key} group={g} />
        ))}
      </div>
    </PanelSection>
  );
}
