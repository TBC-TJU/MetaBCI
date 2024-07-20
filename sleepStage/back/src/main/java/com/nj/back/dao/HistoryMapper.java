package com.nj.back.dao;

import com.nj.back.pojo.History;
import com.nj.back.pojo.Right;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

@Mapper
public interface HistoryMapper {
    public List<History> getHistoryList(int id);

    List<History> getHistoryList1();
}
