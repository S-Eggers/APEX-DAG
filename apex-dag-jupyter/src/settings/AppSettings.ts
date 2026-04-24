import { ISettingRegistry } from '@jupyterlab/settingregistry';

export class AppSettings {
  debounceDelay = 1000;
  replaceDataflowInUDFs = false;
  highlightRelevantSubgraphs = false;
  greedyNotebookExtraction = true;
  llmClassification = false;

  update(settings: ISettingRegistry.ISettings) {
    this.debounceDelay =
      (settings.get('debounceDelay').composite as number) ?? 1000;
    this.replaceDataflowInUDFs =
      (settings.get('replaceDataflowInUDFs').composite as boolean) ?? false;
    this.highlightRelevantSubgraphs =
      (settings.get('highlightRelevantSubgraphs').composite as boolean) ??
      false;
    this.greedyNotebookExtraction =
      (settings.get('greedyNotebookExtraction').composite as boolean) ?? true;
    this.llmClassification =
      (settings.get('llmClassification').composite as boolean) ?? false;
  }
}
