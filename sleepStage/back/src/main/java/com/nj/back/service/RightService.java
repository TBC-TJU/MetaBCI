package com.nj.back.service;

import com.nj.back.pojo.Right;

import java.util.List;

public interface RightService {
    public List<Right> getRightList();

    void updateRightlist(Right right);
}
