import { ServerConnection } from '@jupyterlab/services';
import { GraphMode } from '../types/GraphTypes';

async function callBackend(endpoint: GraphMode, payload: object): Promise<any> {
  const settings = ServerConnection.makeSettings();
  const url = `${settings.baseUrl}apex-dag/${endpoint}`;

  const init: RequestInit = {
    method: 'POST',
    body: JSON.stringify(payload),
    headers: {
      'Content-Type': 'application/json'
    }
  };

  const response = await ServerConnection.makeRequest(url, init, settings);

  if (!response.ok) {
    throw new ServerConnection.ResponseError(response);
  }

  return response.json();
}

export default callBackend;
