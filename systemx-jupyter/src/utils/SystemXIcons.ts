import { LabIcon } from '@jupyterlab/ui-components';

export const SYSTEMX_GRADIENT_SVG = `
<svg style="width: 0; height: 0; position: absolute;" aria-hidden="true" focusable="false">
  <defs>
    <linearGradient id="fancy-grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#6366f1" />
      <stop offset="100%" stop-color="#a855f7" />
    </linearGradient>
  </defs>
</svg>
`;

const SVG_WRAPPER = (content: string) =>
  `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">${content}</svg>`;

const AST_SVG = SVG_WRAPPER(`
  <rect x="3" y="3" width="6" height="6" rx="1" stroke="url(#fancy-grad)" stroke-width="2"/>
  <rect x="15" y="15" width="6" height="6" rx="1" stroke="url(#fancy-grad)" stroke-width="2"/>
  <path d="M9 6H12C13.1046 6 14 6.89543 14 8V16C14 17.1046 14.8954 18 16 18H15" stroke="#CBD5E1" stroke-width="2" stroke-dasharray="2 2"/>
`);

const DATAFLOW_SVG = SVG_WRAPPER(`
  <path d="M12 2L19 21H5L12 2Z" stroke="url(#fancy-grad)" stroke-width="2" stroke-linejoin="round"/>
  <circle cx="12" cy="7" r="2" fill="url(#fancy-grad)" />
  <circle cx="8" cy="15" r="1.5" fill="#CBD5E1" />
  <circle cx="16" cy="15" r="1.5" fill="#CBD5E1" />
`);

const LINEAGE_SVG = SVG_WRAPPER(`
  <path d="M2 12C2 12 5 7 12 7C19 7 22 12 22 12C22 12 19 17 12 17C5 17 2 12 2 12Z" stroke="url(#fancy-grad)" stroke-width="2"/>
  <circle cx="12" cy="12" r="3" stroke="url(#fancy-grad)" stroke-width="2" />
  <path d="M15 12L18 12" stroke="#CBD5E1" stroke-width="2" stroke-linecap="round"/>
`);

const VAMSA_SVG = SVG_WRAPPER(`
  <path d="M4 4L20 20M20 4L4 20" stroke="#CBD5E1" stroke-width="1.5" opacity="0.5"/>
  <rect x="9" y="9" width="6" height="6" rx="1" fill="url(#fancy-grad)" />
  <circle cx="4" cy="4" r="2" fill="url(#fancy-grad)" />
  <circle cx="20" cy="20" r="2" fill="url(#fancy-grad)" />
`);

const LABELING_SVG = SVG_WRAPPER(`
  <path d="M21 15V5C21 3.89543 20.1046 3 19 3H5C3.89543 3 3 3.89543 3 5V15C3 16.1046 3.89543 17 5 17H7L9 20L11 17H19C20.1046 17 21 16.1046 21 15Z" stroke="url(#fancy-grad)" stroke-width="2"/>
  <path d="M7 8H17M7 12H13" stroke="#CBD5E1" stroke-width="2" stroke-linecap="round"/>
`);

const ENV_SVG = SVG_WRAPPER(`
  <path d="M12 2L2 7L12 12L22 7L12 2Z" fill="url(#fancy-grad)" fill-opacity="0.2" stroke="url(#fancy-grad)" stroke-width="2"/>
  <path d="M2 12L12 17L22 12" stroke="#CBD5E1" stroke-width="2" stroke-linecap="round"/>
  <path d="M2 17L12 22L22 17" stroke="#CBD5E1" stroke-width="2" stroke-linecap="round"/>
`);

const EXECUTION_STATE_SVG = SVG_WRAPPER(`
  <circle cx="6" cy="5" r="2" fill="url(#fancy-grad)" />
  <circle cx="6" cy="12" r="2" fill="url(#fancy-grad)" />
  <circle cx="6" cy="19" r="2" fill="#CBD5E1" />
  <path d="M6 7V10M6 14V17" stroke="url(#fancy-grad)" stroke-width="2" stroke-linecap="round"/>
  <path d="M11 5H19M11 12H19M11 19H16" stroke="#CBD5E1" stroke-width="2" stroke-linecap="round"/>
  <path d="M18 16L21 19L18 22" stroke="url(#fancy-grad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
`);

const SETTINGS_SVG = SVG_WRAPPER(`
  <circle cx="12" cy="12" r="3" stroke="url(#fancy-grad)" stroke-width="2" />
  <path d="M12 2V4M12 20V22M4 12H2M22 12H20M5.6 5.6L7 7M17 17L18.4 18.4M18.4 5.6L17 7M7 17L5.6 18.4" stroke="#CBD5E1" stroke-width="2" stroke-linecap="round"/>
`);

const LEAKAGE_SVG = SVG_WRAPPER(`
  <path d="M10.29 3.86L1.82 18C1.47 18.63 1.51 19.4 1.93 19.99C2.29 20.5 2.88 20.8 3.51 20.8H20.49C21.12 20.8 21.71 20.5 22.07 19.99C22.49 19.4 22.53 18.63 22.18 18L13.71 3.86C13.35 3.24 12.69 2.86 11.98 2.86C11.28 2.86 10.65 3.24 10.29 3.86Z" stroke="url(#fancy-grad)" stroke-width="2" stroke-linejoin="round"/>
  <path d="M12 9V13" stroke="#CBD5E1" stroke-width="2" stroke-linecap="round"/>
  <circle cx="12" cy="16.5" r="1" fill="#CBD5E1"/>
`);

export const astIcon = new LabIcon({ name: 'systemx:ast', svgstr: AST_SVG });
export const dataflowIcon = new LabIcon({
  name: 'systemx:dataflow',
  svgstr: DATAFLOW_SVG
});
export const lineageIcon = new LabIcon({
  name: 'systemx:lineage',
  svgstr: LINEAGE_SVG
});
export const vamsaIcon = new LabIcon({ name: 'systemx:vamsa', svgstr: VAMSA_SVG });
export const labelingIcon = new LabIcon({
  name: 'systemx:labeling',
  svgstr: LABELING_SVG
});
export const environmentIcon = new LabIcon({
  name: 'systemx:environment',
  svgstr: ENV_SVG
});
export const executionStateIcon = new LabIcon({
  name: 'systemx:execution-state',
  svgstr: EXECUTION_STATE_SVG
});
const EXECUTION_TRACE_SVG = SVG_WRAPPER(`
  <circle cx="14" cy="10" r="7" stroke="url(#fancy-grad)" stroke-width="2"/>
  <path d="M14 6V10L17 12" stroke="url(#fancy-grad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="4" cy="6" r="1.5" fill="#CBD5E1" />
  <circle cx="4" cy="12" r="1.5" fill="#CBD5E1" />
  <circle cx="4" cy="18" r="1.5" fill="url(#fancy-grad)" />
  <path d="M8 18H14M14 18L12 16M14 18L12 20" stroke="#CBD5E1" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
`);
export const executionTraceIcon = new LabIcon({
  name: 'systemx:execution-trace',
  svgstr: EXECUTION_TRACE_SVG
});
export const settingsIcon = new LabIcon({
  name: 'systemx:settings',
  svgstr: SETTINGS_SVG
});
export const leakageIcon = new LabIcon({
  name: 'systemx:leakage',
  svgstr: LEAKAGE_SVG
});
const TUPLE_ANNOTATION_SVG = SVG_WRAPPER(`
  <circle cx="5" cy="7" r="2.5" stroke="url(#fancy-grad)" stroke-width="2"/>
  <circle cx="15" cy="15" r="2.5" stroke="url(#fancy-grad)" stroke-width="2"/>
  <path d="M7 8.5L13 13.5" stroke="#CBD5E1" stroke-width="2" stroke-linecap="round"/>
  <path d="M20 5V11M17 8H23" stroke="url(#fancy-grad)" stroke-width="2" stroke-linecap="round"/>
`);
export const tupleAnnotationIcon = new LabIcon({
  name: 'systemx:tuple-annotation',
  svgstr: TUPLE_ANNOTATION_SVG
});
