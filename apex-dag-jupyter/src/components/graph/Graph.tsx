import React, {useEffect, useRef, useState} from "react";
import cytoscape from "cytoscape";
import dagre from "cytoscape-dagre";


cytoscape.use(dagre);

export default function Graph({ graphData = {elements: []} }) {
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
    
    const edgeType = (element: cytoscape.SingularElementReturnValue) => {
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

    const nodeType = (element: cytoscape.SingularElementReturnValue) => {
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

    const lineType = (element: cytoscape.SingularElementReturnValue) => {
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
            "background-color": (element: cytoscape.SingularElementReturnValue) => nodeType(element),
            "label": "data(label)",
            "width": "60px",
            "height": "35px",
            "text-valign": "center" as "center" | "top" | "bottom",
            "text-halign": "center" as "center" | "left" | "right",
            "font-size": "12px",
            "color": "#333"
            }
        },
        {
            selector: "edge",
            style: {
                "width": 2,
                "line-color": (element: cytoscape.SingularElementReturnValue) => edgeType(element),
                "target-arrow-shape": "triangle",
                "target-arrow-color": (element: cytoscape.SingularElementReturnValue) => edgeType(element),
                "curve-style": "bezier",
                "label": "data(label)",
                "line-style": (element: cytoscape.SingularElementReturnValue) => lineType(element) as "solid" | "dashed",
            }
        }
    ];

    const graphRef = useRef(null);
    const [pan, setPan] = useState<cytoscape.Position | null>(null);
    const [zoom, setZoom] = useState(1);

    const drawGraph = () => {
        const cy = cytoscape({
            container: graphRef.current,
            style: style,
            layout: layout,
            elements: graphData.elements,
        });
        
        if (pan) {
            cy.pan(pan);
        } else {
            console.log("Centering the graph");
            cy.center();
        }
        cy.zoom(zoom);
        cy.on("pan", () => {
            setPan(cy.pan());
        });

        cy.on("zoom", () => {
            setZoom(cy.zoom());
        });

        setPan(cy.pan());
        setZoom(cy.zoom());
    };
       
    useEffect(() => {
        drawGraph()
    }, [graphData]);

    return (
        <>
            <div id="cy" className={"cy"} ref={graphRef}></div>
            <ul className={"legend"}>
                {legendItems.map((item, index) => (
                <li key={index}>
                    <div className={item.type} style={{ backgroundColor: item.color, borderColor: item.color, borderStyle: item.borderStyle }}></div> {item.label}
                </li>
                ))}
            </ul>
        </>
    );
}
