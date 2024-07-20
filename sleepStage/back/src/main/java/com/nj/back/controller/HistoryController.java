package com.nj.back.controller;

import com.nj.back.pojo.History;
import com.nj.back.pojo.Right;
import com.nj.back.service.HistoryService;
import com.nj.back.service.RightService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/adminapi/history")
public class HistoryController {
    @Autowired
    private HistoryService historyService;
    @GetMapping(value = "/{id}")
    public List<History> getHistoryList(@PathVariable Integer id){
        return historyService.getHistoryList(id);
    }
    @GetMapping()
    public List<History> getHistoryList(){
        return historyService.getHistoryList();
    }
}
