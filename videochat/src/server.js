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

function onSocketClose() {
    console.log("Disconnected from the Browser ❌");
  }
  
const sockets = [];
  
wss.on("connection", (socket) => {
    sockets.push(socket);
    socket["nickname"] = "unKnown";
    console.log("Connected to Browser ✅");
    socket.on("close", onSocketClose);
    socket.on("message", (msg) => {
        const message = JSON.parse(msg);
        switch (message.type){
            case "new_message":
                sockets.forEach((aSocket) =>
                  aSocket.send(`${socket.nickname}: ${message.payload}`)
                );
                break
            case "nickname":
                socket["nickname"] = message.payload;
                break
   
        }
    });
  });

server.listen(3000, handleListen);
