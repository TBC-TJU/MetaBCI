import socket

soc = socket.socket()
soc.connect(('127.0.0.1',65432))

while True:
    msg = input("请输入发送给服务端的消息：")
    if "exit" == msg:
        break
    soc.send(msg.encode("UTF-8"))
    # data = soc.recv(1024).decode("UTF-8")
    # print(f"服务端发来的消息：{data}")
soc.close()
