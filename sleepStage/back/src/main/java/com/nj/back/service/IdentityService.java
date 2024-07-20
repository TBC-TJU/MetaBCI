package com.nj.back.service;

import com.nj.back.pojo.Identity;
import jakarta.annotation.Resource;
import org.springframework.stereotype.Service;


//登录认证
@Service
public class IdentityService {
    @Resource
    private IdentityMapper identityMapper;

    public Identity LoginIn(String username, String password) {
        return identityMapper.getInfo(username,password);
    }

    public void Insert(String username,String password){
        identityMapper.saveInfo(username, password);
    }
}
