import { ISettingRegistry } from '@jupyterlab/settingregistry';

export class AppSettings {
  rawDatasetPath = 'raw_dataset';
  greedyNotebookExtraction = true;
  replaceDataflowInUDFs = false;
  highlightRelevantSubgraphs = false;
  useGraphRefiner = true;
  llmClassification = false;
  debounceDelay = 1000;

  update(settings: ISettingRegistry.ISettings) {
    const extraction = (settings.get('extraction').composite as any) || {};
    this.rawDatasetPath = extraction.rawDatasetPath ?? 'raw_dataset';
    this.greedyNotebookExtraction = extraction.greedyNotebookExtraction ?? true;

    const processing = (settings.get('processing').composite as any) || {};
    this.replaceDataflowInUDFs = processing.replaceDataflowInUDFs ?? false;
    this.highlightRelevantSubgraphs =
      processing.highlightRelevantSubgraphs ?? false;
    this.useGraphRefiner = processing.useGraphRefiner ?? true;

    const ml = (settings.get('ml').composite as any) || {};
    this.llmClassification = ml.llmClassification ?? false;

    const ui = (settings.get('ui').composite as any) || {};
    this.debounceDelay = ui.debounceDelay ?? -1;
  }
}
