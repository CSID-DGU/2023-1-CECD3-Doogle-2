import http from "http";
import WebSocket from "ws";
import express from "express";
import path from 'path';

const __dirname = path.resolve();
const app = express();

app.set("view engine", "pug");
app.set("views", path.join(__dirname, "src", "public", "views")); // 뷰 디렉토리 설정
app.use("/public", express.static(path.join(__dirname, "src", "public")));
app.get("/", (req, res) => res.render("home"));
app.get("/*", (req, res) => res.redirect("/"));

const handleListen = () => console.log(`Listening on http://localhost:3000`);

const server = http.createServer(app);

const wss = new WebSocket.Server({server});

wss.on("connection", (socket) => {
    console.log("Connected to Browser ✅");
    socket.on("close", () => console.log("Disconnected from the Browser ❌"));
    socket.on("message", (message) => {
      console.log(message);
    });
    socket.send("hello!!!");
  });

server.listen(3000, handleListen);


