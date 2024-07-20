package com.nj.back.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;

@Configuration
public class UDPConfig {
    @Bean
    public OPStreamUDPClient opStreamUDPClient() throws UnknownHostException, SocketException {
        return new OPStreamUDPClient(InetAddress.getLocalHost().getHostAddress(),"12345");
    }
}
