import http from "http";
import WebSocket from "ws";
import express from "express";
import path from 'path';
import SocketIO from "socket.io";

const __dirname = path.resolve();
const app = express();

app.set("view engine", "pug");
app.set("views", path.join(__dirname, "src", "public", "views")); // 뷰 디렉토리 설정
app.use("/public", express.static(path.join(__dirname, "src", "public")));
app.get("/", (req, res) => res.render("home"));
app.get("/*", (req, res) => res.redirect("/"));

const httpServer = http.createServer(app);
const wsServer = SocketIO(httpServer);

wsServer.on("connection", (socket) => {
    socket.on("join_room", (roomName) => {
      socket.join(roomName);
      socket.to(roomName).emit("welcome");
    });
    socket.on("offer", (offer, roomName) => {
        socket.to(roomName).emit("offer", offer);
      });
      socket.on("answer", (answer, roomName) => {
        socket.to(roomName).emit("answer", answer);
      });
      socket.on("ice", (ice, roomName) => {
        socket.to(roomName).emit("ice", ice);
      });
  });

const handleListen = () => console.log(`Listening on http://localhost:3000`);
httpServer.listen(3000, handleListen);
