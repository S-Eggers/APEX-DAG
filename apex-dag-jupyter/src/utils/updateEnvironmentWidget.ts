import { NotebookPanel } from '@jupyterlab/notebook';
import { EnvironmentWidget } from '../components/widget/EnvironmentWidget';
import callBackend from './callBackend';
import { getNotebookCode } from './getNotebookCode';

export const updateEnvironmentWidget = (
  widget: EnvironmentWidget | null,
  notebookPanel: NotebookPanel,
  settings: any // Replace with your AppSettings interface
) => {
  if (!widget) return;

  const cells = notebookPanel.content.model?.cells;
  if (!cells) {
    console.warn('No notebook cells available for environment extraction.');
    return;
  }

  const content = getNotebookCode(cells, settings.greedyNotebookExtraction);
  console.debug('[ENVIRONMENT] Extracted notebook content for telemetry.');

  const payload = {
    code: content
  };

  callBackend('environment', payload)
    .then(response => {
      console.info('[ENVIRONMENT] Raw response received:', response);

      const parsedResponse =
        typeof response === 'string' ? JSON.parse(response) : response;

      if (!parsedResponse.success) {
        console.error(
          '[ENVIRONMENT] Backend returned failure.',
          parsedResponse
        );
        return;
      }

      if (parsedResponse.environment_data) {
        console.debug(
          '[ENVIRONMENT] Successfully parsed telemetry, updating widget.'
        );
        widget.updateEnvironmentData(parsedResponse.environment_data);
      } else {
        console.error(
          '[ENVIRONMENT] Payload is missing the "environment_data" object.',
          parsedResponse
        );
      }
    })
    .catch(error => {
      console.error('[ENVIRONMENT] Network or parsing error:', error);
    });
};

export default updateEnvironmentWidget;
