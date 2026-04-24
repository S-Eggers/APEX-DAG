import { Menu, Widget } from '@lumino/widgets';
import { caretDownEmptyIcon } from '@jupyterlab/ui-components';
import { WIDGET_REGISTRY } from '../../registry/WidgetRegistry';

export class ApexNativeDropdownWidget extends Widget {
  constructor(commands: any) {
    super({ node: document.createElement('div') });
    this.addClass('jp-Toolbar-item');

    const button = document.createElement('button');
    button.className = 'jp-ToolbarButtonComponent';
    button.title = 'Open APEX-DAG Views';

    button.style.display = 'flex';
    button.style.alignItems = 'center';
    button.style.gap = '4px';
    button.style.cursor = 'pointer';
    button.style.border = 'none';
    button.style.background = 'transparent';
    button.style.padding = '0px 8px';
    button.style.height = '100%';

    button.addEventListener('mouseenter', () => {
      button.style.backgroundColor = 'var(--jp-layout-color2)';
    });
    button.addEventListener('mouseleave', () => {
      button.style.backgroundColor = 'transparent';
    });

    const label = document.createElement('span');
    label.className = 'jp-ToolbarButtonComponent-label';
    label.innerText = 'APEX-DAG';
    button.appendChild(label);

    const iconSpan = document.createElement('span');
    iconSpan.className = 'jp-ToolbarButtonComponent-icon';
    caretDownEmptyIcon.element({
      container: iconSpan,
      width: '16px',
      height: '16px'
    });
    button.appendChild(iconSpan);

    this.node.appendChild(button);

    button.onclick = () => {
      const menu = new Menu({ commands });

      let currentGroup = -1;

      WIDGET_REGISTRY.forEach(config => {
        if (currentGroup !== -1 && config.group !== currentGroup) {
          menu.addItem({ type: 'separator' });
        }
        currentGroup = config.group;

        menu.addItem({
          command: config.commandId,
          args: { isToolbar: true }
        });
      });

      const rect = button.getBoundingClientRect();
      menu.open(rect.left, rect.bottom);
    };
  }
}
