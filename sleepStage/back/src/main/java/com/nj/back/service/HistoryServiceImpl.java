package com.nj.back.service;

import com.nj.back.dao.HistoryMapper;
import com.nj.back.dao.RightMapper;
import com.nj.back.pojo.History;
import com.nj.back.pojo.Right;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class HistoryServiceImpl implements HistoryService{
    @Autowired
    private HistoryMapper historyMapper;
    public List<History> getHistoryList(int id){
        return historyMapper.getHistoryList(id);
    }

    @Override
    public List<History> getHistoryList() {
        return historyMapper.getHistoryList1();
    }


}
