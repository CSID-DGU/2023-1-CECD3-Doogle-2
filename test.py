import cv2
import socket
import pickle
import struct

# 캠 초기화
cap = cv2.VideoCapture(0)

# 소켓 초기화
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('210.94.194.83', 8080))  # 원격 서버의 IP 주소와 포트 번호

while True:
    # 캠에서 프레임 읽기
    ret, frame = cap.read()

    # 프레임을 바이트로 변환하여 소켓으로 전송
    data = pickle.dumps(frame)
    try:
        client_socket.sendall(struct.pack("Q", len(data)) + data)
    except (BrokenPipeError, ConnectionResetError):
        print("Connection closed. Exiting...")
        break

