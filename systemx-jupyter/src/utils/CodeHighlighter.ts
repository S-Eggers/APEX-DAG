import { Extension, StateEffect, StateField } from '@codemirror/state';
import { syntaxTree } from '@codemirror/language';
import { Decoration, DecorationSet } from '@codemirror/view';
import { Cell } from '@jupyterlab/cells';
import { CodeMirrorEditor } from '@jupyterlab/codemirror';
import { NotebookPanel } from '@jupyterlab/notebook';

export interface HighlightTarget {
  cellId: string;
  codeText: string;
  color: string;
  domainLabel: string;
  nodeType?: number;
  nodeId?: string;
  wholeWord?: boolean;
}

function _escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

interface _SyntaxNodeLike {
  type: { name: string };
  parent: _SyntaxNodeLike | null;
}

function _isInStringOrComment(state: unknown, pos: number): boolean {
  let node = syntaxTree(state as Parameters<typeof syntaxTree>[0])
    .resolveInner(pos, 1) as unknown as _SyntaxNodeLike | null;
  for (; node; node = node.parent) {
    const name = node.type.name;
    if (name === 'FormatReplacement') return false;
    if (name === 'String' || name === 'FormatString' || name === 'Comment') {
      return true;
    }
  }
  return false;
}

interface ICMView {
  dispatch: (...args: unknown[]) => void;
  state: {
    doc: { toString(): string };
    field<T>(field: StateField<T>, require?: false): T | undefined;
  };
}

interface _MarkRange {
  from: number;
  to: number;
  color: string;
  label: string;
  nodeId?: string;
}

const _setMarks = StateEffect.define<_MarkRange[]>();
const _clearMarks = StateEffect.define<void>();

const _marksField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update: (deco, tr) => {
    deco = deco.map(tr.changes);
    for (const e of tr.effects) {
      if (e.is(_clearMarks)) {
        deco = Decoration.none;
      }
      if (e.is(_setMarks)) {
        const marks = e.value.map(r => {
          const styleProps = [
            'text-decoration: underline wavy',
            `text-decoration-color: ${r.color}`,
            'text-decoration-thickness: 2px',
            'text-underline-offset: 3px'
          ];
          if (r.nodeId) styleProps.push('cursor: pointer');
          const attrs: Record<string, string> = {
            style: styleProps.join('; '),
            'aria-label': `SystemX: ${r.label}`,
            'data-systemx-label': r.label,
            'data-systemx-color': r.color
          };
          if (r.nodeId) {
            attrs['data-systemx-node-id'] = r.nodeId;
            attrs['data-systemx-clickable'] = 'true';
          }
          return Decoration.mark({ attributes: attrs }).range(r.from, r.to);
        });
        deco = deco.update({ add: marks, sort: true });
      }
    }
    return deco;
  },
  provide: f => {
    const { EditorView } = require('@codemirror/view') as any;
    return EditorView.decorations.from(f);
  }
});

const _marksExtension: Extension = _marksField;

const _injectedEditors = new WeakSet<CodeMirrorEditor>();

function _getCMEditor(cell: Cell): CodeMirrorEditor | null {
  const editor = cell.editor;
  if (!editor) return null;
  return editor as CodeMirrorEditor;
}

function _getView(cmEditor: CodeMirrorEditor): ICMView {
  return cmEditor.editor as unknown as ICMView;
}

function _ensureExtension(cmEditor: CodeMirrorEditor): void {
  if (_injectedEditors.has(cmEditor)) return;
  cmEditor.injectExtension(_marksExtension);
  _injectedEditors.add(cmEditor);
}

export function applyHighlights(
  nbPanel: NotebookPanel | null,
  targets: HighlightTarget[]
): void {
  if (!nbPanel?.content?.widgets) return;

  const byCell = new Map<string, HighlightTarget[]>();
  for (const t of targets) {
    if (!t.cellId || !t.codeText?.trim()) continue;
    const list = byCell.get(t.cellId) ?? [];
    list.push(t);
    byCell.set(t.cellId, list);
  }

  for (const cellWidget of nbPanel.content.widgets) {
    const cmEditor = _getCMEditor(cellWidget);
    if (!cmEditor) continue;

    _ensureExtension(cmEditor);
    const view = _getView(cmEditor);

    view.dispatch({ effects: _clearMarks.of(undefined) });

    const cellTargets = byCell.get(cellWidget.model.id);
    if (!cellTargets?.length) continue;

    const state = view.state;
    const docText = view.state.doc.toString();
    const ranges: _MarkRange[] = [];

    for (const t of cellTargets) {
      if (t.wholeWord) {
        const re = new RegExp(`\\b${_escapeRegExp(t.codeText)}\\b`, 'g');
        let m: RegExpExecArray | null;
        while ((m = re.exec(docText)) !== null) {
          if (!_isInStringOrComment(state, m.index)) {
            ranges.push({
              from: m.index,
              to: m.index + t.codeText.length,
              color: t.color,
              label: t.domainLabel,
              nodeId: t.nodeId
            });
          }
          if (m.index === re.lastIndex) re.lastIndex++; // guard zero-width
        }
      } else {
        let idx = 0;
        while ((idx = docText.indexOf(t.codeText, idx)) !== -1) {
          if (!_isInStringOrComment(state, idx)) {
            ranges.push({
              from: idx,
              to: idx + t.codeText.length,
              color: t.color,
              label: t.domainLabel,
              nodeId: t.nodeId
            });
          }
          idx += t.codeText.length;
        }
      }
    }

    if (ranges.length > 0) {
      view.dispatch({ effects: _setMarks.of(ranges) });
    }
  }
}

export function clearHighlights(nbPanel: NotebookPanel | null): void {
  if (!nbPanel?.content?.widgets) return;
  for (const cellWidget of nbPanel.content.widgets) {
    const cmEditor = _getCMEditor(cellWidget);
    if (!cmEditor) continue;
    if (_injectedEditors.has(cmEditor)) {
      const view = _getView(cmEditor);
      view.dispatch({ effects: _clearMarks.of(undefined) });
    }
  }
}

const _clickHandlers = new Map<Element, EventListener>();

export function registerHighlightClickHandler(
  nbPanel: NotebookPanel | null,
  onNodeClick: (nodeId: string) => void
): void {
  if (!nbPanel?.node) return;

  unregisterHighlightClickHandler(nbPanel);

  const handler: EventListener = evt => {
    const target = evt.target as HTMLElement;
    const span = target.closest('[data-systemx-node-id]') as HTMLElement | null;
    if (!span) return;
    const nodeId = span.getAttribute('data-systemx-node-id');
    if (nodeId) onNodeClick(nodeId);
  };

  nbPanel.node.addEventListener('click', handler);
  _clickHandlers.set(nbPanel.node, handler);
}

export function unregisterHighlightClickHandler(
  nbPanel: NotebookPanel | null
): void {
  if (!nbPanel?.node) return;
  const handler = _clickHandlers.get(nbPanel.node);
  if (handler) {
    nbPanel.node.removeEventListener('click', handler);
    _clickHandlers.delete(nbPanel.node);
  }
}

let _tooltipEl: HTMLDivElement | null = null;

function _ensureTooltip(): HTMLDivElement {
  if (_tooltipEl) return _tooltipEl;

  const el = document.createElement('div');
  el.setAttribute('role', 'tooltip');
  Object.assign(el.style, {
    position: 'fixed',
    zIndex: '10000',
    display: 'none',
    alignItems: 'center',
    gap: '8px',
    padding: '5px 9px',
    borderRadius: '7px',
    background: '#21262d',
    border: '1px solid #30363d',
    boxShadow: '0 6px 18px rgb(0 0 0 / 38%)',
    font: '500 12px/1.2 var(--jp-ui-font-family, system-ui, sans-serif)',
    color: '#e8eaed',
    whiteSpace: 'nowrap',
    pointerEvents: 'none',
    opacity: '0',
    transition: 'opacity 90ms ease'
  } as Partial<CSSStyleDeclaration>);

  const dot = document.createElement('span');
  dot.dataset.role = 'dot';
  Object.assign(dot.style, {
    width: '9px',
    height: '9px',
    borderRadius: '50%',
    flex: '0 0 auto'
  } as Partial<CSSStyleDeclaration>);

  const label = document.createElement('span');
  label.dataset.role = 'label';

  const chip = document.createElement('span');
  chip.textContent = 'SystemX';
  Object.assign(chip.style, {
    padding: '1px 6px',
    borderRadius: '999px',
    background: 'rgb(99 102 241 / 92%)',
    color: '#fff',
    fontSize: '9px',
    fontWeight: '700',
    letterSpacing: '0.04em'
  } as Partial<CSSStyleDeclaration>);

  el.append(dot, label, chip);
  document.body.appendChild(el);
  _tooltipEl = el;
  return el;
}

function _showTooltip(target: HTMLElement): void {
  const labelText = target.getAttribute('data-systemx-label');
  if (!labelText) return;

  const el = _ensureTooltip();
  const dot = el.querySelector<HTMLElement>('[data-role="dot"]');
  const label = el.querySelector<HTMLElement>('[data-role="label"]');
  if (dot) dot.style.background = target.getAttribute('data-systemx-color') || '#6366f1';
  if (label) label.textContent = labelText;

  el.style.display = 'flex';
  const r = target.getBoundingClientRect();
  const t = el.getBoundingClientRect();
  let top = r.top - t.height - 6;
  if (top < 4) top = r.bottom + 6;
  const left = Math.max(
    4,
    Math.min(r.left + r.width / 2 - t.width / 2, window.innerWidth - t.width - 4)
  );
  el.style.top = `${Math.round(top)}px`;
  el.style.left = `${Math.round(left)}px`;
  requestAnimationFrame(() => {
    el.style.opacity = '1';
  });
}

function _hideTooltip(): void {
  if (!_tooltipEl) return;
  _tooltipEl.style.opacity = '0';
  _tooltipEl.style.display = 'none';
}

interface _HoverPair {
  over: EventListener;
  out: EventListener;
}
const _hoverHandlers = new Map<Element, _HoverPair>();

export function registerHighlightHoverHandler(
  nbPanel: NotebookPanel | null
): void {
  if (!nbPanel?.node) return;

  unregisterHighlightHoverHandler(nbPanel);

  const over: EventListener = evt => {
    const span = (evt.target as HTMLElement)?.closest?.(
      '[data-systemx-label]'
    ) as HTMLElement | null;
    if (span) _showTooltip(span);
  };
  const out: EventListener = evt => {
    const span = (evt.target as HTMLElement)?.closest?.('[data-systemx-label]');
    if (span) _hideTooltip();
  };

  nbPanel.node.addEventListener('mouseover', over);
  nbPanel.node.addEventListener('mouseout', out);
  _hoverHandlers.set(nbPanel.node, { over, out });
}

export function unregisterHighlightHoverHandler(
  nbPanel: NotebookPanel | null
): void {
  if (!nbPanel?.node) return;
  const pair = _hoverHandlers.get(nbPanel.node);
  if (pair) {
    nbPanel.node.removeEventListener('mouseover', pair.over);
    nbPanel.node.removeEventListener('mouseout', pair.out);
    _hoverHandlers.delete(nbPanel.node);
  }
  _hideTooltip();
}
