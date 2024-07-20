package com.nj.back.dao;

import com.nj.back.pojo.Right;
import lombok.Data;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface RightMapper {
//    @Select("select * from rights")
    public List<Right> getRightList();

    void updateRightList(Right right);
}
