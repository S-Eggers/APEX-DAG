import { NotebookPanel } from '@jupyterlab/notebook';
import { ExecutionTraceWidget } from '../components/widget/ExecutionTraceWidget';
import callBackend from './callBackend';
import { getNotebookCode } from './getNotebookCode';
import { TraceAnalysisData } from '../types/ExecutionTraceTypes';
import { AppSettings } from '../settings/AppSettings';
import { traceRecorder } from './ExecutionTraceRecorder';

export const updateExecutionTraceWidget = (
  widget: ExecutionTraceWidget | null,
  notebookPanel: NotebookPanel,
  settings: AppSettings
): Promise<void> => {
  if (!widget) return Promise.resolve();

  const cells = notebookPanel.content.model?.cells;
  if (!cells) return Promise.resolve();

  const path = notebookPanel.context.path;
  const trace = traceRecorder.getTrace(path);

  const payload = {
    cells: getNotebookCode(cells, true),
    trace: { sessions: trace.sessions, sources: trace.sources },
    filename: path,
    detectDsl: settings.detectDsl,
    execBackend: settings.execOrderBackend
  };

  return callBackend<{ success: boolean; execution_trace: TraceAnalysisData }>(
    'execution-trace/analyze',
    payload
  )
    .then(response => {
      if (!response.success || !response.execution_trace) {
        console.error('[EXECUTION-TRACE] Backend returned failure.', response);
        return;
      }
      widget.updateAnalysis(response.execution_trace);
      widget.trackNotebook(notebookPanel);
      traceRecorder.setCellGraph(path, response.execution_trace.cell_graph);
    })
    .catch(error => {
      console.error('[EXECUTION-TRACE] Network or parsing error:', error);
    });
};

export default updateExecutionTraceWidget;
