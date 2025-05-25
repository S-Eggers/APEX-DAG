const chokidar = require("chokidar");
const WebSocketWS = require("ws");
const fs = require("fs");

const filePath = process.argv[2];

if (!filePath) {
  console.error("Please provide a file path as an argument");
  process.exit(1);
}
const wss = new WebSocketWS.Server({ port: 8081 });

console.log("WebSocket server running ws://localhost:8081");

wss.on("connection", (ws) => {
  console.log("Client connected");
  if (fs.existsSync(filePath)) {
    ws.send(JSON.stringify({ type: "update", data: JSON.parse(fs.readFileSync(filePath, "utf8")) }));
  }
});

chokidar.watch(filePath).on("change", () => {
  setTimeout(() => {
    console.log("File changed, sending update to clients");
    const data = JSON.parse(fs.readFileSync(filePath, "utf8"));
    wss.clients.forEach(client => {
      if (client.readyState === WebSocketWS.OPEN) {
        client.send(JSON.stringify({ type: "update", data }));
      }
    });
  }, 50);
});