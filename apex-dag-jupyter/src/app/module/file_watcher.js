const chokidar = require("chokidar");
const WebSocketWS = require("ws");
const fs = require("fs");

const filePath = "../data_flow_graph.json"; // Pfad zur JSON-Datei
const wss = new WebSocketWS.Server({ port: 8081 });

console.log("WebSocket server running ws://localhost:8081");

wss.on("connection", (ws) => {
  if (fs.existsSync(filePath)) {
    ws.send(JSON.stringify({ type: "update", data: JSON.parse(fs.readFileSync(filePath, "utf8")) }));
  }
});

chokidar.watch(filePath).on("change", () => {
  setTimeout(() => {
    const data = JSON.parse(fs.readFileSync(filePath, "utf8"));
    wss.clients.forEach(client => {
      if (client.readyState === WebSocketWS.OPEN) {
        client.send(JSON.stringify({ type: "update", data }));
      }
    });
  }, 50);
});