import http from "http";
import WebSocket from "ws";
import express from "express";
import path from 'path';
import SocketIO from "socket.io";
import fs from 'fs'; // Node.js의 파일 시스템 모듈 추가
import request from 'request-promise';

const __dirname = path.resolve();
const app = express();

app.set("view engine", "pug");
app.set("views", path.join(__dirname, "src", "public", "views"));
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

const uploadDir = path.join(__dirname, 'uploads'); // 업로드된 파일을 저장할 디렉토리
fs.mkdirSync(uploadDir, { recursive: true }); // 디렉토리가 없으면 생성

const multer = require('multer');

const storage = multer.memoryStorage(); // 메모리에 저장

const upload = multer({ storage: storage });

app.post('/uploads', upload.single('video'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file provided" });
  }

  const videoBuffer = req.file.buffer;
  
  // Python 서버로 파일을 전송
  try {
      const response = await sendVideo(videoBuffer);
      console.log(response);
      res.json({ message: "File uploaded and sent successfully" });
  } catch (error) {
      console.error(error);
      res.status(500).json({ error: "Failed to send file to Python server" });
  }
});

async function sendVideo(videoBuffer) {
  // 파일을 Python 서버로 전송
  const options = {
      method: 'POST',
      uri: 'http://127.0.0.1:5000/sendVideo',
      body: {
          video: videoBuffer
      },
      json: true
  };

  try {
      const response = await request(options);
      return response;
  } catch (error) {
      throw error;
  }
}


const handleListen = () => console.log(`Listening on http://localhost:3000`);
httpServer.listen(3000, handleListen);
