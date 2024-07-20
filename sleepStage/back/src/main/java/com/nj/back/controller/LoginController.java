package com.nj.back.controller;

import com.nj.back.pojo.Identity;
import com.nj.back.service.IdentityService;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;


@Slf4j
@RestController
@RequestMapping("/adminapi")
public class LoginController {

    //将Service注入Web层
    @Resource
    IdentityService identityService;

    //实现登录
    @RequestMapping("/login")
    public String login(String username,String password){
        Identity identityBean = identityService.LoginIn(username, password);
        log.info("name:{}",username);
        log.info("password:{}",password);
        if(identityBean!=null){
            return "success";
        }else {
            return "error";
        }
    }

    //实现注册功能
    @RequestMapping("/register")
    public String signUp(String username,String password){
        identityService.Insert(username, password);
        return "success";
    }
}
