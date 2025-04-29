import { LabIcon } from '@jupyterlab/ui-components';

import svgContent from '../style/apex-dag-logo.svg';

// Create a LabIcon from it
const apexDagLogo = new LabIcon({
  name: 'apex:logo',
  svgstr: svgContent
});

export default apexDagLogo;