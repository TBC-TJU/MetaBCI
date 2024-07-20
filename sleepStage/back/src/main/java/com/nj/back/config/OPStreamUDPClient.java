package com.nj.back.config;

import java.net.*;
import java.util.Arrays;
import java.util.Base64;

public class OPStreamUDPClient {

    private int remotePort;  // UDP服务器端口
    private InetAddress remoteIP;  // UDP服务器地址
    private DatagramSocket socket;  // UDPSocket
    private final DatagramSocket receiveSocket = new DatagramSocket(23456);




    // 构造方法初始化所需变量
    public OPStreamUDPClient(String remoteIp, String remotePort) throws SocketException {
        try{
            this.remoteIP = InetAddress.getByName(remoteIp);
            this.remotePort = Integer.parseInt(remotePort);
            socket = new DatagramSocket();
        }catch (Exception e){
            e.printStackTrace();
        }

    }

    // 定义一个数据的发送方法
    public void send(String msg){
        try {
            // 将待发送的字符串转为字节数组
            byte[] outData = msg.getBytes("utf-8");
            // 构建用于发送的数据报文，构造方法中传入远程通信方（服务器）的IP地址和端口
            DatagramPacket outPacket = new DatagramPacket(outData, outData.length, remoteIP, remotePort);
            // 给UDP服务器发送数据报
            socket.send(outPacket);
        }
        catch (Exception e){
            e.printStackTrace();
        }

    }

    public String receive(DatagramPacket receivePacket) throws Exception {
//        InetAddress inetAddress = receiveSocket.getInetAddress();
//        System.out.println();
        // 准备空的数据报文
        receiveSocket.receive(receivePacket);
        String flightInfo=new String(receivePacket.getData());
        flightInfo=flightInfo.trim();//通过trim去掉空字符
        return flightInfo;
    }


}