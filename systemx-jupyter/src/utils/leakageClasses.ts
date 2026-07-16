import { LeakageGoldEntry } from '../types/GraphTypes';

export function valueToGold(
  entries: LeakageGoldEntry[],
  value: number
): string {
  return entries[value]?.key ?? entries[0]?.key ?? 'clean';
}

export function goldToValue(
  entries: LeakageGoldEntry[],
  gold: string | undefined
): number {
  const idx = entries.findIndex(entry => entry.key === gold);
  return idx >= 0 ? idx : 0;
}
