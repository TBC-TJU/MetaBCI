package com.nj.back.service;

import com.nj.back.dao.HistoryMapper;
import com.nj.back.dao.UserMapper;
import com.nj.back.pojo.History;
import com.nj.back.pojo.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class UserServiceImpl implements UserService{
    @Autowired
    private UserMapper userMapper;
    public List<User> getUserList(){
        return userMapper.getUserList();
    }
}
