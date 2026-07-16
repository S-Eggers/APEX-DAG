import { getBackend } from './callBackend';
import {
  GraphMode,
  TaxonomyAPIResponse,
  TaxonomyModeData
} from '../types/GraphTypes';

export interface EdgeMeta {
  color: string;
  label: string;
}

export type MetaResolver = (code: number) => EdgeMeta | undefined;

export interface HighlightResolvers {
  operation: MetaResolver;
  entity: MetaResolver;
}

let _taxonomyCache: Promise<Record<string, TaxonomyModeData>> | null = null;

function loadAllTaxonomies(): Promise<Record<string, TaxonomyModeData>> {
  if (!_taxonomyCache) {
    _taxonomyCache = getBackend<TaxonomyAPIResponse>('constants')
      .then(res => (res.success && res.taxonomy ? res.taxonomy : {}))
      .catch((err: Error) => {
        _taxonomyCache = null;
        console.error('Failed to load taxonomy constants', err.message);
        return {};
      });
  }
  return _taxonomyCache;
}

function makeResolver(table: Record<string, { color: string; label: string }>): MetaResolver {
  return (code: number) => {
    const meta = table[code.toString()];
    return meta ? { color: meta.color, label: meta.label } : undefined;
  };
}

export async function loadHighlightResolvers(
  mode: GraphMode
): Promise<HighlightResolvers> {
  const all = await loadAllTaxonomies();
  const modeTaxonomy = all[mode];
  return {
    operation: makeResolver(
      modeTaxonomy?.hubs ?? modeTaxonomy?.edges ?? {}
    ),
    entity: makeResolver(modeTaxonomy?.nodes ?? {})
  };
}
