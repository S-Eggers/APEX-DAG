"use strict";
(self["webpackChunkapex_dag"] = self["webpackChunkapex_dag"] || []).push([["lib_index_js"],{

/***/ "./lib/index.js":
/*!**********************!*\
  !*** ./lib/index.js ***!
  \**********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "default": () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyterlab/apputils */ "webpack/sharing/consume/default/@jupyterlab/apputils");
/* harmony import */ var _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _jupyterlab_launcher__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @jupyterlab/launcher */ "webpack/sharing/consume/default/@jupyterlab/launcher");
/* harmony import */ var _jupyterlab_launcher__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_launcher__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @jupyterlab/ui-components */ "webpack/sharing/consume/default/@jupyterlab/ui-components");
/* harmony import */ var _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _widget__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./widget */ "./lib/widget.js");




var CommandIDs;
(function (CommandIDs) {
    CommandIDs.create = 'apex-dag-widget';
})(CommandIDs || (CommandIDs = {}));
/**
 * Initialization data for the apex-dag extension.
 */
const plugin = {
    id: 'apex-dag:plugin',
    description: 'APEX-DAG Jupyter Lab Extension',
    autoStart: true,
    optional: [_jupyterlab_launcher__WEBPACK_IMPORTED_MODULE_1__.ILauncher],
    activate: (app, launcher) => {
        const { commands } = app;
        const command = CommandIDs.create;
        commands.addCommand(command, {
            caption: 'APEX-DAG',
            label: 'APEX-DAG Widget',
            icon: args => (args['isPalette'] ? undefined : _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_2__.reactIcon),
            execute: () => {
                const content = new _widget__WEBPACK_IMPORTED_MODULE_3__.GraphWidget();
                const widget = new _jupyterlab_apputils__WEBPACK_IMPORTED_MODULE_0__.MainAreaWidget({ content });
                widget.title.label = 'APEX-DAG';
                widget.title.icon = _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_2__.reactIcon;
                app.shell.add(widget, 'main');
            }
        });
        if (launcher) {
            launcher.add({
                command
            });
        }
    }
};
/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (plugin);


/***/ }),

/***/ "./lib/widget.js":
/*!***********************!*\
  !*** ./lib/widget.js ***!
  \***********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   GraphWidget: () => (/* binding */ GraphWidget)
/* harmony export */ });
/* harmony import */ var _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @jupyterlab/ui-components */ "webpack/sharing/consume/default/@jupyterlab/ui-components");
/* harmony import */ var _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ "webpack/sharing/consume/default/react");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var cytoscape__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! cytoscape */ "../node_modules/cytoscape/dist/cytoscape.esm.mjs");
/* harmony import */ var cytoscape_dagre__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! cytoscape-dagre */ "../node_modules/cytoscape-dagre/cytoscape-dagre.js");
/* harmony import */ var cytoscape_dagre__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(cytoscape_dagre__WEBPACK_IMPORTED_MODULE_3__);




cytoscape__WEBPACK_IMPORTED_MODULE_2__["default"].use((cytoscape_dagre__WEBPACK_IMPORTED_MODULE_3___default()));
/**
 * React component for a counter.
 *
 * @returns The React component
 */
const GraphComponent = ({ graphData = { elements: [] } }) => {
    const layout = {
        name: "dagre",
        rankDir: "TB",
    };
    const colors = {
        light_steel_blue: "#B0C4DE",
        very_soft_blue: "#b3b0de",
        pink: "#FFC0CB",
        light_green: "#c4deb0",
        very_soft_yellow: "#DEDAB0",
        very_soft_purple: "#DEB0DE",
        very_soft_lime_green: "#B0DEB9",
        light_salmon: "#FFA07A",
        pale_green: "#98FB98",
        gray: "#d3d3d3",
        powder_blue: "#B0E0E6",
        peach_puff: "#FFDAB9",
    };
    const legendItems = [
        { type: "node", color: colors.light_steel_blue, label: "Variable", borderStyle: "solid" },
        { type: "node", color: colors.very_soft_blue, label: "Intermediate", borderStyle: "solid" },
        { type: "node", color: colors.light_green, label: "Function", borderStyle: "solid" },
        { type: "node", color: colors.pink, label: "Import", borderStyle: "solid" },
        { type: "node", color: colors.very_soft_yellow, label: "If", borderStyle: "solid" },
        { type: "node", color: colors.very_soft_lime_green, label: "Loop", borderStyle: "solid" },
        { type: "node", color: colors.very_soft_purple, label: "Class", borderStyle: "solid" },
        { type: "edge", color: colors.light_salmon, label: "Caller", borderStyle: "solid" },
        { type: "edge", color: colors.gray, label: "Reassign", borderStyle: "dashed" },
        { type: "edge", color: colors.pale_green, label: "Input", borderStyle: "solid" },
        { type: "edge", color: colors.powder_blue, label: "Branch", borderStyle: "solid" },
        { type: "edge", color: colors.peach_puff, label: "Loop", borderStyle: "solid" },
        { type: "edge", color: colors.light_green, label: "Function", borderStyle: "solid" }
    ];
    const edgeType = (element) => {
        const caseType = element.data("edge_type");
        switch (caseType) {
            case 0: return colors.light_salmon;
            case 1: return colors.pale_green;
            case 2: return colors.gray;
            case 3: return colors.powder_blue;
            case 4: return colors.peach_puff;
            case 5: return colors.light_green;
            default: return "#000";
        }
    };
    const nodeType = (element) => {
        const caseType = element.data("node_type");
        switch (caseType) {
            case 0: return colors.light_steel_blue;
            case 1: return colors.pink;
            /*case 2: return colors.light_green; <- not used yet*/
            case 3: return colors.light_green;
            case 4: return colors.very_soft_blue;
            case 5: return colors.very_soft_yellow;
            case 6: return colors.very_soft_lime_green;
            case 7: return colors.very_soft_purple;
            default: return "#000";
        }
    };
    const lineType = (element) => {
        const caseType = element.data("edge_type");
        switch (caseType) {
            case 2: return "dashed";
            default: return "solid";
        }
    };
    const style = [{
            selector: "node",
            style: {
                "shape": "round-rectangle",
                "background-color": (element) => nodeType(element),
                "label": "data(label)",
                "width": "60px",
                "height": "35px",
                "text-valign": "center",
                "text-halign": "center",
                "font-size": "12px",
                "color": "#333"
            }
        },
        {
            selector: "edge",
            style: {
                "width": 2,
                "line-color": (element) => edgeType(element),
                "target-arrow-shape": "triangle",
                "target-arrow-color": (element) => edgeType(element),
                "curve-style": "bezier",
                "label": "data(label)",
                "line-style": (element) => lineType(element),
            }
        }
    ];
    const graphRef = (0,react__WEBPACK_IMPORTED_MODULE_1__.useRef)(null);
    const [pan, setPan] = (0,react__WEBPACK_IMPORTED_MODULE_1__.useState)(null);
    const [zoom, setZoom] = (0,react__WEBPACK_IMPORTED_MODULE_1__.useState)(1);
    const drawGraph = () => {
        const cy = (0,cytoscape__WEBPACK_IMPORTED_MODULE_2__["default"])({
            container: graphRef.current,
            style: style,
            layout: layout,
            elements: graphData.elements,
        });
        if (pan) {
            cy.pan(pan);
        }
        else {
            console.log("Centering the graph");
            cy.center();
        }
        cy.zoom(zoom);
        cy.on('pan', () => {
            setPan(cy.pan());
        });
        cy.on('zoom', () => {
            setZoom(cy.zoom());
        });
        setPan(cy.pan());
        setZoom(cy.zoom());
    };
    (0,react__WEBPACK_IMPORTED_MODULE_1__.useEffect)(() => {
        drawGraph();
    }, [graphData]);
    return (react__WEBPACK_IMPORTED_MODULE_1___default().createElement((react__WEBPACK_IMPORTED_MODULE_1___default().Fragment), null,
        react__WEBPACK_IMPORTED_MODULE_1___default().createElement("div", { id: "cy", ref: graphRef }),
        react__WEBPACK_IMPORTED_MODULE_1___default().createElement("ul", { className: "legend" }, legendItems.map((item, index) => (react__WEBPACK_IMPORTED_MODULE_1___default().createElement("li", { key: index },
            react__WEBPACK_IMPORTED_MODULE_1___default().createElement("div", { className: item.type, style: { backgroundColor: item.color, borderColor: item.color, borderStyle: item.borderStyle } }),
            " ",
            item.label))))));
};
/**
 * A widget that displays a graph.
 */
class GraphWidget extends _jupyterlab_ui_components__WEBPACK_IMPORTED_MODULE_0__.ReactWidget {
    constructor() {
        super();
        this.addClass('jp-react-widget');
    }
    render() {
        return react__WEBPACK_IMPORTED_MODULE_1___default().createElement(GraphComponent, null);
    }
}


/***/ })

}]);
//# sourceMappingURL=lib_index_js.1c50dca4ba82a695f3ac.js.map