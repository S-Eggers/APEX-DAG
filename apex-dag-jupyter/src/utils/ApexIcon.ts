import { LabIcon } from "@jupyterlab/ui-components";

import svgContent from "../../style/apex-dag-logo.svg";


// Create a LabIcon from it
const ApexIcon = new LabIcon({
  name: "apex:logo",
  svgstr: svgContent
});

export default ApexIcon;