import { ISettingRegistry } from '@jupyterlab/settingregistry';

export type NNBackend = 'hgt' | 'mlp' | 'xgboost' | 'vamsa_static';

export type FeaturePreset =
  | 'standard'
  | 'all'
  | 'emb_only'
  | 'api_lib'
  | 'struct_only';

interface IExtractionSettings {
  rawDatasetPath?: string;
  greedyNotebookExtraction?: boolean;
  defaultFetchMode?: 'unannotated' | 'annotated' | 'flagged';
  defaultFlagType?: 'bug_in_dataflow' | 'needs_review' | 'not_relevant';
}

interface IProcessingSettings {
  replaceDataflowInUDFs?: boolean;
  highlightRelevantSubgraphs?: boolean;
  useGraphRefiner?: boolean;
  usePredictionForAnnotation?: boolean;
  explainFeatureImportance?: boolean;
  detectDsl?: boolean;
}

export type ExecOrderBackend =
  | 'heuristic'
  | 'exec_order_mlp'
  | 'exec_order_xgboost';

interface IMLSettings {
  nnBackend?: NNBackend;
  featurePreset?: FeaturePreset;
  modelVariant?: string;
  execOrderBackend?: ExecOrderBackend;
}

interface IUISettings {
  debounceDelay?: number;
}

export type TraceAutosaveMode = 'off' | 'debounced' | 'session-end';

interface IExecutionTraceSettings {
  enabled?: boolean;
  autosave?: TraceAutosaveMode;
  autosaveDelay?: number;
  storagePath?: string;
}

export class AppSettings {
  rawDatasetPath = 'raw_dataset';
  greedyNotebookExtraction = true;

  defaultFetchMode: 'unannotated' | 'annotated' | 'flagged' = 'unannotated';
  defaultFlagType: string = 'needs_review';

  replaceDataflowInUDFs = false;
  highlightRelevantSubgraphs = false;
  useGraphRefiner = true;
  detectDsl = false;
  nnBackend: NNBackend = 'hgt';
  featurePreset: FeaturePreset = 'standard';
  modelVariant = '';
  execOrderBackend: ExecOrderBackend = 'heuristic';
  usePredictionForAnnotation = false;
  explainFeatureImportance = false;
  debounceDelay = 1000;

  traceEnabled = true;
  traceAutosave: TraceAutosaveMode = 'debounced';
  traceAutosaveDelay = 5000;
  traceStoragePath = 'systemx_traces';

  update(settings: ISettingRegistry.ISettings): void {
    const extraction =
      (settings.get('extraction')
        .composite as unknown as IExtractionSettings) || {};
    this.rawDatasetPath = extraction.rawDatasetPath ?? 'raw_dataset';
    this.greedyNotebookExtraction = extraction.greedyNotebookExtraction ?? true;
    this.defaultFetchMode = extraction.defaultFetchMode ?? 'unannotated';
    this.defaultFlagType = extraction.defaultFlagType ?? 'needs_review';

    const processing =
      (settings.get('processing')
        .composite as unknown as IProcessingSettings) || {};
    this.replaceDataflowInUDFs = processing.replaceDataflowInUDFs ?? false;
    this.detectDsl = processing.detectDsl ?? false;
    this.highlightRelevantSubgraphs =
      processing.highlightRelevantSubgraphs ?? false;
    this.useGraphRefiner = processing.useGraphRefiner ?? true;
    this.usePredictionForAnnotation =
      processing.usePredictionForAnnotation ?? false;
    this.explainFeatureImportance =
      processing.explainFeatureImportance ?? false;

    const ml = (settings.get('ml').composite as unknown as IMLSettings) || {};
    this.nnBackend = ml.nnBackend ?? 'hgt';
    this.featurePreset = ml.featurePreset ?? 'standard';
    this.modelVariant = ml.modelVariant ?? '';
    this.execOrderBackend = ml.execOrderBackend ?? 'heuristic';

    const ui = (settings.get('ui').composite as unknown as IUISettings) || {};
    this.debounceDelay = ui.debounceDelay ?? 1000;

    const trace =
      (settings.get('executionTrace')
        .composite as unknown as IExecutionTraceSettings) || {};
    this.traceEnabled = trace.enabled ?? true;
    this.traceAutosave = trace.autosave ?? 'debounced';
    this.traceAutosaveDelay = trace.autosaveDelay ?? 5000;
    this.traceStoragePath = trace.storagePath ?? 'systemx_traces';
  }
}
