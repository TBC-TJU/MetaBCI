package com.nj.back.dao;

import com.nj.back.pojo.File;
import org.apache.ibatis.annotations.Mapper;


@Mapper
public interface FileMapper {

    void updateFile(File file);
}
