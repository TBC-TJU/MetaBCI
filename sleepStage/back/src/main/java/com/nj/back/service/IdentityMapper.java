package com.nj.back.service;


import com.nj.back.pojo.Identity;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface IdentityMapper {
    //查询，可以实现登录功能
    @Select("SELECT * FROM identity WHERE name = #{username} AND password = #{password}")
    Identity getInfo(@Param("username") String username, @Param("password") String password);

    //多个参数要加@Param修饰
    //增加，可以实现注册功能
    @Insert("insert into identity(name,password)values(#{username},#{password})")
    void saveInfo(@Param("username") String username, @Param("password") String password);
}

