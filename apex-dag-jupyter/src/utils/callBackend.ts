import { ServerConnection } from '@jupyterlab/services';
import { GraphMode, TableMode } from '../types/GraphTypes';

export async function callBackend(
  endpoint: GraphMode | TableMode | string,
  payload: object
): Promise<any> {
  const settings = ServerConnection.makeSettings();

  let route = endpoint as string;
  if (endpoint === 'labeling') {
    route = 'labeling/predict';
  }
  if (endpoint === 'vamsa_wir') {
    route = 'vamsa/wir';
  }
  if (endpoint === 'vamsa_lineage') {
    route = 'vamsa/lineage';
  }

  const url = `${settings.baseUrl}apex-dag/${route}`;

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

export async function getBackend(endpoint: string): Promise<any> {
  const settings = ServerConnection.makeSettings();
  const url = `${settings.baseUrl}apex-dag/${endpoint}`;

  const init: RequestInit = { method: 'GET' };
  const response = await ServerConnection.makeRequest(url, init, settings);

  if (!response.ok) throw new ServerConnection.ResponseError(response);
  return response.json();
}

export default callBackend;
