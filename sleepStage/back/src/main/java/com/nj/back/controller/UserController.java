package com.nj.back.controller;

import com.nj.back.pojo.User;
import com.nj.back.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/adminapi/user")
public class UserController {
    @Autowired
    private UserService userService;
    @GetMapping
    public List<User> getUserList(){
        return userService.getUserList();
    }
}
