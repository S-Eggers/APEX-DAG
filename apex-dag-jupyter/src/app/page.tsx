"use client"

import { useEffect, useState } from "react";
import Graph from "./module/graph";

export default function Home() {

  const [graphData, setGraphData] = useState({ elements: [] });

  useEffect(() => {
    const socket = new WebSocket(process.env.REACT_APP_WEBSOCKET_URL || "ws://localhost:8000");

    socket.onmessage = (event) => {
      const newGraphData = JSON.parse(event.data);
      setGraphData(newGraphData);
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
