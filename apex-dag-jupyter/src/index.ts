import { JupyterFrontEnd, JupyterFrontEndPlugin} from "@jupyterlab/application";
import { ICommandPalette, MainAreaWidget } from "@jupyterlab/apputils";
import { ISettingRegistry } from "@jupyterlab/settingregistry";
import { ILauncher } from "@jupyterlab/launcher";
import { INotebookTracker, NotebookPanel } from "@jupyterlab/notebook";
import { IMainMenu } from "@jupyterlab/mainmenu";

import { GraphWidget } from "./components/widget/GraphWidget";
import ApexIcon from "./utils/ApexIcon"
import updateWidget from "./utils/updateWidget";
import updateLineageWidget from "./utils/updateLineageWidget";
import CommandIDs from "./types/CommandIDs";


/**
 * Initialization data for the apex-dag-jupyter extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: CommandIDs.plugin,
  description: "APEX-DAG Jupyter Frontend Extension.",
  autoStart: true,
  optional: [ISettingRegistry],
  requires: [ICommandPalette, INotebookTracker, ILauncher, IMainMenu],
  activate: (app: JupyterFrontEnd, palette: ICommandPalette, tracker: INotebookTracker, launcher: ILauncher, mainMenu: IMainMenu) => {
    console.debug("JupyterLab extension apex-dag-jupyter is activated!");
    console.debug("ICommandPalette:", palette);

    const dataflowWidget: string = CommandIDs.dataflow;
    let dataflowGraphWidget: GraphWidget;
    app.commands.addCommand(dataflowWidget, {
      caption: "Dataflow Widget",
      label: "Dataflow Widget",
      icon: args => (args["isPalette"] ? undefined : ApexIcon),
      execute: () => {
        const content = new GraphWidget();
        const widget = new MainAreaWidget<GraphWidget>({ content });
        widget.title.label = "Dataflow Widget";
        widget.title.icon = ApexIcon;
        widget.title.closable = true;
        app.shell.add(widget, "main");
        dataflowGraphWidget = content;
      }
    });

    const dataflowCommandItem = {
      command: dataflowWidget,
      category: "APEX-DAG",
      rank: 1,
    }

    const lineageWidget: string = CommandIDs.lineage;
    let lineageGraphWidget: GraphWidget;
    app.commands.addCommand(lineageWidget, {
      caption: "Lineage Widget",
      label: "Lineage Widget",
      icon: args => (args["isPalette"] ? undefined : ApexIcon),
      execute: () => {
        const content = new GraphWidget();
        const widget = new MainAreaWidget<GraphWidget>({ content });
        widget.title.label = "Lineage Widget";
        widget.title.icon = ApexIcon;
        widget.title.closable = true;
        app.shell.add(widget, "main");
        lineageGraphWidget = content;
      }
    });

    const lineageCommandItem = {
      command: lineageWidget,
      category: "APEX-DAG",
      rank: 2,
    }

    palette.addItem(dataflowCommandItem);
    palette.addItem(lineageCommandItem);
    mainMenu.fileMenu.newMenu.addGroup([dataflowCommandItem, lineageCommandItem]);
    launcher.add(dataflowCommandItem);
    launcher.add(lineageCommandItem);

    tracker.currentChanged.connect(async (sender, notebookPanel: NotebookPanel | null) => {
      console.debug("Current notebook changed", notebookPanel);
      if (!notebookPanel) {
        console.warn("No active notebook or dataflow widget.");
        return;
      }
  
      await notebookPanel.revealed;
      console.debug("Notebook revealed");
      const content = notebookPanel.content;
      const model = content.model;
      if (!content || !model) {
        console.error("NotebookPanel has no content or model.");
        return;
      }

      console.debug("Initial call to updateWidget on notebook open");
      updateWidget(dataflowGraphWidget, notebookPanel);
      updateLineageWidget(lineageGraphWidget, notebookPanel);

      console.debug("Setting up listener for model changes");
      model.contentChanged.connect(() => {
        console.debug("Notebook model changed!");
        updateWidget(dataflowGraphWidget, notebookPanel);
        updateLineageWidget(lineageGraphWidget, notebookPanel);
      });
    });
  }
};

export default plugin;
