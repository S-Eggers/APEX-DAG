import { ServerConnection } from '@jupyterlab/services';
import { GraphMode, TableMode } from '../types/GraphTypes';

export async function callBackend<T = unknown>(
  endpoint: GraphMode | TableMode | string,
  payload: Record<string, unknown>
): Promise<T> {
  const settings = ServerConnection.makeSettings();

  let route = endpoint as string;
  if (endpoint === 'labeling') {
    route = 'labeling/predict';
  } else if (endpoint === 'vamsa_wir') {
    route = 'vamsa/wir';
  } else if (endpoint === 'vamsa_lineage') {
    route = 'vamsa/lineage';
  }

  const url = `${settings.baseUrl}systemx/${route}`;

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

  return response.json() as Promise<T>;
}

export async function getBackend<T = unknown>(endpoint: string): Promise<T> {
  const settings = ServerConnection.makeSettings();
  const url = `${settings.baseUrl}systemx/${endpoint}`;

  const init: RequestInit = { method: 'GET' };
  const response = await ServerConnection.makeRequest(url, init, settings);

  if (!response.ok) {
    throw new ServerConnection.ResponseError(response);
  }

  return response.json() as Promise<T>;
}

export default callBackend;
