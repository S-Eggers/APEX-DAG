"use client"

import { useEffect, useState } from "react";
import Graph from "./module/graph";

export default function Home() {

  const [graphData, setGraphData] = useState({ elements: [] });
  //setGraphData(require("./data_flow_graph.json"));

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8081");

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "update") {
        const newGraphData = message.data;
        console.log(newGraphData);
        console.log(typeof newGraphData);
        setGraphData(newGraphData);
      }
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    return () => {
      socket.close();
    };
    
  }, []);

  return (
    <>
      <div className={"page"}>
        <Graph graphData={graphData} />
      </div>
    </>
  );
}
