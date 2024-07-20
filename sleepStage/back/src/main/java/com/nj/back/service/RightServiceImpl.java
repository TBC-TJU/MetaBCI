package com.nj.back.service;

import com.nj.back.dao.RightMapper;
import com.nj.back.pojo.Right;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

//左侧功能栏

@Service
public class RightServiceImpl implements RightService{

    @Autowired
    private RightMapper rightMapper;
    public List<Right> getRightList(){
        return rightMapper.getRightList();
    }

    @Override
    public void updateRightlist(Right right) {
        rightMapper.updateRightList(right);
    }
}
