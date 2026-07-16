import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { useDarkMode } from '../../hooks/useDarkMode';
import { getBackend } from '../../utils/callBackend';

type FieldSchema = {
  type?: string;
  title?: string;
  description?: string;
  enum?: string[];
  enumDescriptions?: string[];
  minimum?: number;
};

type GroupSchema = {
  type?: string;
  title?: string;
  description?: string;
  properties?: Record<string, FieldSchema>;
  default?: Record<string, unknown>;
};

interface IDiscoveredDataset {
  path: string;
  label: string;
  notebooks: number;
  annotations: number;
  has_annotations: boolean;
}

interface IModelVariant {
  key: string;
  family: string;
  preset: string | null;
  base_key: string | null;
  created_at: string | null;
  metrics: { train_accuracy?: number; num_call_nodes?: number };
  is_finetuned: boolean;
  loaded: boolean;
}

interface SystemXSettingsComponentProps {
  settings: ISettingRegistry.ISettings;
}

const DISABLED_WHEN: Record<
  string,
  (group: Record<string, unknown>) => string | null
> = {
  'ml.featurePreset': g =>
    g.nnBackend === 'vamsa_static'
      ? 'Not used by the vamsa_static model (static KB lookup, no learned features).'
      : null,
  'extraction.defaultFlagType': g =>
    g.defaultFetchMode !== 'flagged'
      ? 'Only used when Default Labeling Mode is "flagged".'
      : null
};

const onKeyActivate =
  (fn: () => void) => (e: React.KeyboardEvent<HTMLElement>) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fn();
    }
  };

function Toggle({
  checked,
  disabled,
  onChange,
  label
}: {
  checked: boolean;
  disabled?: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  const toggle = () => {
    if (!disabled) onChange(!checked);
  };
  return (
    <div
      role="switch"
      aria-checked={checked}
      aria-label={label}
      aria-disabled={disabled}
      tabIndex={disabled ? -1 : 0}
      onClick={toggle}
      onKeyDown={onKeyActivate(toggle)}
      className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-400 ${
        disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'
      } ${checked ? 'bg-indigo-500' : 'bg-gray-300 dark:bg-gray-600'}`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
          checked ? 'translate-x-4' : 'translate-x-0.5'
        }`}
      />
    </div>
  );
}

const INPUT_CLASS =
  'mt-0.5 w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-[#161b22] px-2 py-1.5 text-[13px] text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-400 disabled:cursor-not-allowed';

function Field({
  fieldKey,
  schema,
  value,
  disabledReason,
  suggestions,
  variantOptions,
  onChange
}: {
  fieldKey: string;
  schema: FieldSchema;
  value: unknown;
  disabledReason: string | null;
  suggestions?: IDiscoveredDataset[];
  variantOptions?: { value: string; label: string }[];
  onChange: (v: unknown) => void;
}) {
  const disabled = disabledReason !== null;
  const label = schema.title ?? fieldKey;
  const hasSuggestions = suggestions !== undefined;
  const hasVariantOptions = variantOptions !== undefined;
  const isBool = schema.type === 'boolean';
  const isEnum =
    !hasSuggestions &&
    !hasVariantOptions &&
    Array.isArray(schema.enum) &&
    schema.enum.length > 0;
  const isNumber = schema.type === 'number' || schema.type === 'integer';
  const listId = `systemx-set-list-${fieldKey}`;

  const selectedEnumHelp =
    isEnum && schema.enumDescriptions
      ? schema.enumDescriptions[schema.enum!.indexOf(String(value))]
      : undefined;

  return (
    <div
      className={`flex flex-col gap-1 py-2.5 ${disabled ? 'opacity-60' : ''}`}
    >
      <div className="flex items-start justify-between gap-3">
        <label className="text-[13px] font-medium text-gray-800 dark:text-gray-200">
          {label}
        </label>
        {isBool && (
          <Toggle
            checked={Boolean(value)}
            disabled={disabled}
            onChange={onChange}
            label={label}
          />
        )}
      </div>

      {hasSuggestions && (
        <>
          <input
            type="text"
            aria-label={label}
            list={listId}
            disabled={disabled}
            value={String(value ?? '')}
            onChange={e => onChange(e.target.value)}
            className={`${INPUT_CLASS} font-mono`}
          />
          <datalist id={listId}>
            {suggestions!.map(d => (
              <option key={d.path} value={d.path}>
                {`${d.label} - ${d.notebooks} notebooks${
                  d.has_annotations ? `, ${d.annotations} annotations` : ''
                }`}
              </option>
            ))}
          </datalist>
        </>
      )}

      {isEnum && (
        <select
          aria-label={label}
          disabled={disabled}
          value={String(value ?? '')}
          onChange={e => onChange(e.target.value)}
          className={INPUT_CLASS}
        >
          {schema.enum!.map(opt => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      )}

      {hasVariantOptions && (
        <select
          aria-label={label}
          disabled={disabled}
          value={String(value ?? '')}
          onChange={e => onChange(e.target.value)}
          className={INPUT_CLASS}
        >
          {variantOptions!.map(opt => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      )}

      {!hasSuggestions && isNumber && (
        <input
          type="number"
          aria-label={label}
          disabled={disabled}
          min={schema.minimum}
          value={Number.isFinite(value as number) ? (value as number) : ''}
          onChange={e =>
            onChange(e.target.value === '' ? 0 : Number(e.target.value))
          }
          className={`${INPUT_CLASS} w-32`}
        />
      )}

      {!hasSuggestions &&
        !hasVariantOptions &&
        !isBool &&
        !isEnum &&
        !isNumber && (
          <input
            type="text"
            aria-label={label}
            disabled={disabled}
            value={String(value ?? '')}
            onChange={e => onChange(e.target.value)}
            className={`${INPUT_CLASS} font-mono`}
          />
        )}

      {schema.description && (
        <p className="text-[11px] leading-snug text-gray-500 dark:text-gray-400">
          {schema.description}
        </p>
      )}
      {hasSuggestions && suggestions!.length > 0 && (
        <p className="text-[11px] leading-snug text-indigo-500 dark:text-indigo-300">
          {suggestions!.length} dataset
          {suggestions!.length > 1 ? 's' : ''} auto-discovered under ./data -
          pick from the list or type a custom path.
        </p>
      )}
      {selectedEnumHelp && (
        <p className="text-[11px] leading-snug text-indigo-500 dark:text-indigo-300">
          {selectedEnumHelp}
        </p>
      )}
      {disabledReason && (
        <p className="text-[11px] leading-snug text-amber-600 dark:text-amber-400">
          {disabledReason}
        </p>
      )}
    </div>
  );
}

const slug = (s: string): string =>
  `systemx-set-${s.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}`;

export default function SystemXSettingsComponent({
  settings
}: SystemXSettingsComponentProps) {
  const isDark = useDarkMode();
  const [, forceRender] = useState(0);
  const [query, setQuery] = useState('');
  const [activeGroup, setActiveGroup] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<IDiscoveredDataset[] | null>(null);
  const [variants, setVariants] = useState<IModelVariant[] | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const cb = () => forceRender(v => v + 1);
    settings.changed.connect(cb);
    return () => {
      settings.changed.disconnect(cb);
    };
  }, [settings]);

  useEffect(() => {
    let mounted = true;
    getBackend<{ success: boolean; datasets: IDiscoveredDataset[] }>('datasets')
      .then(res => {
        if (mounted) setDatasets(res?.datasets ?? []);
      })
      .catch(() => {
        if (mounted) setDatasets([]);
      });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    const load = () => {
      getBackend<{ success: boolean; variants: IModelVariant[] }>('models')
        .then(res => {
          if (mounted) setVariants(res?.variants ?? []);
        })
        .catch(() => {
          if (mounted) setVariants([]);
        });
    };
    load();
    settings.changed.connect(load);
    return () => {
      mounted = false;
      settings.changed.disconnect(load);
    };
  }, [settings]);

  const variantOptions = useMemo(() => {
    const opts: { value: string; label: string }[] = [
      { value: '', label: 'Base - use Model + Feature Set above' }
    ];
    for (const v of (variants ?? []).filter(x => x.is_finetuned)) {
      const acc = v.metrics?.train_accuracy;
      opts.push({
        value: v.key,
        label: `${v.key}${acc != null ? ` · train acc ${acc}` : ''}`
      });
    }
    return opts;
  }, [variants]);

  const groups = useMemo(() => {
    const props = (settings.schema.properties ?? {}) as Record<
      string,
      GroupSchema
    >;
    return Object.entries(props)
      .filter(([, g]) => g.type === undefined || g.properties)
      .map(([key, schema]) => ({ key, schema }));
  }, [settings]);

  const groupValue = (groupKey: string): Record<string, unknown> =>
    (settings.get(groupKey).composite ?? {}) as Record<string, unknown>;

  const setField = (groupKey: string, fieldKey: string, value: unknown) => {
    const next = { ...groupValue(groupKey), [fieldKey]: value };
    void settings.set(groupKey, next as never);
  };

  const restoreDefaults = () => {
    for (const { key, schema } of groups) {
      void settings.set(key, (schema.default ?? {}) as never);
    }
  };

  const q = query.trim().toLowerCase();
  const fieldMatches = (fieldKey: string, f: FieldSchema): boolean => {
    if (!q) return true;
    return (
      fieldKey.toLowerCase().includes(q) ||
      (f.title ?? '').toLowerCase().includes(q) ||
      (f.description ?? '').toLowerCase().includes(q)
    );
  };

  useEffect(() => {
    const root = scrollRef.current;
    if (!root) return;
    const observer = new IntersectionObserver(
      entries => {
        const visible = entries
          .filter(e => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActiveGroup(visible[0].target.id);
      },
      { root, rootMargin: '0px 0px -70% 0px', threshold: 0 }
    );
    root
      .querySelectorAll('[data-group-card]')
      .forEach(el => observer.observe(el));
    return () => observer.disconnect();
  }, [groups, q]);

  const scrollToGroup = (key: string) => {
    const el = scrollRef.current?.querySelector(`#${slug(key)}`);
    el?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    setActiveGroup(slug(key));
  };

  return (
    <div
      className={`${isDark ? 'dark ' : ''}flex flex-col h-full w-full box-border bg-[#f9fbfe] dark:bg-[#0d1117] text-gray-800 dark:text-gray-200`}
    >
      {/* Header */}
      <div className="shrink-0 flex flex-wrap items-center gap-3 px-5 py-3 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-[#161b22]">
        <div className="flex flex-col mr-auto">
          <h1 className="text-sm font-bold tracking-wide">SystemX Settings</h1>
          <span className="text-[11px] text-gray-500 dark:text-gray-400">
            Configuration for the SystemX JupyterLab extension.
          </span>
        </div>
        <input
          type="search"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Search settings..."
          aria-label="Search settings"
          className="w-48 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-[#0d1117] px-2.5 py-1.5 text-[13px] focus:outline-none focus:ring-2 focus:ring-indigo-400"
        />
        <div
          role="button"
          tabIndex={0}
          onClick={restoreDefaults}
          onKeyDown={onKeyActivate(restoreDefaults)}
          className="cursor-pointer select-none rounded bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 px-3 py-1.5 text-[12px] font-medium text-gray-700 dark:text-gray-200 border border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-indigo-400"
        >
          Restore Defaults
        </div>
      </div>

      <div className="flex flex-1 min-h-0">
        <nav className="w-44 shrink-0 overflow-y-auto overflow-x-hidden border-r border-gray-200 dark:border-gray-700 py-3 hidden sm:block">
          {groups.map(({ key, schema }) => {
            const anyMatch = Object.entries(schema.properties ?? {}).some(
              ([fk, f]) => fieldMatches(fk, f)
            );
            if (!anyMatch) return null;
            const id = slug(key);
            return (
              <div
                key={key}
                role="button"
                tabIndex={0}
                onClick={() => scrollToGroup(key)}
                onKeyDown={onKeyActivate(() => scrollToGroup(key))}
                className={`block w-full text-left px-4 py-1.5 text-[12px] cursor-pointer select-none transition-colors border-l-2 focus:outline-none ${
                  activeGroup === id
                    ? 'border-indigo-500 text-indigo-600 dark:text-indigo-300 bg-indigo-50 dark:bg-indigo-900/20 font-semibold'
                    : 'border-transparent text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
                }`}
              >
                {schema.title ?? key}
              </div>
            );
          })}
        </nav>

        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-5 flex flex-col gap-5"
        >
          {groups.map(({ key, schema }) => {
            const fields = Object.entries(schema.properties ?? {}).filter(
              ([fk, f]) => fieldMatches(fk, f)
            );
            if (fields.length === 0) return null;
            const gv = groupValue(key);
            return (
              <section
                key={key}
                id={slug(key)}
                data-group-card
                className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#161b22] shadow-sm"
              >
                <header className="px-4 py-2.5 border-b border-gray-100 dark:border-gray-700">
                  <h2 className="text-[13px] font-bold tracking-wide text-gray-800 dark:text-gray-200">
                    {schema.title ?? key}
                  </h2>
                </header>
                <div className="px-4 py-1 divide-y divide-gray-100 dark:divide-gray-800">
                  {fields.map(([fk, f]) => {
                    const rule = DISABLED_WHEN[`${key}.${fk}`];
                    const disabledReason = rule ? rule(gv) : null;
                    const suggestions =
                      key === 'extraction' && fk === 'rawDatasetPath'
                        ? (datasets ?? [])
                        : undefined;
                    const varOpts =
                      key === 'ml' && fk === 'modelVariant'
                        ? variantOptions
                        : undefined;
                    return (
                      <Field
                        key={fk}
                        fieldKey={fk}
                        schema={f}
                        value={gv[fk]}
                        disabledReason={disabledReason}
                        suggestions={suggestions}
                        variantOptions={varOpts}
                        onChange={v => setField(key, fk, v)}
                      />
                    );
                  })}
                </div>
              </section>
            );
          })}
          {groups.every(({ schema }) =>
            Object.entries(schema.properties ?? {}).every(
              ([fk, f]) => !fieldMatches(fk, f)
            )
          ) && (
            <div className="text-center text-sm text-gray-400 dark:text-gray-500 py-10">
              No settings match "{query}".
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
