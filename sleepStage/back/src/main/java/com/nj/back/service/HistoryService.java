package com.nj.back.service;

import com.nj.back.pojo.History;

import java.util.List;

public interface HistoryService {
    public List<History> getHistoryList(int id);

    List<History> getHistoryList();
}
