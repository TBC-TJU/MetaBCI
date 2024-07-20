package com.nj.back.dao;

import com.nj.back.pojo.User;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

@Mapper
public interface UserMapper {
    List<User> getUserList();
}
