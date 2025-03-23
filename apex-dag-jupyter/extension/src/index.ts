import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

/**
 * Initialization data for the apex-dag extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'apex-dag:plugin',
  description: 'APEX-DAG Jupyter Lab Extension',
  autoStart: true,
  activate: (app: JupyterFrontEnd) => {
    console.log('JupyterLab extension apex-dag is activated!');
  }
};

export default plugin;
