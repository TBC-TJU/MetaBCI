package com.nj.back.controller;

import com.nj.back.pojo.Right;
import com.nj.back.service.RightService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/adminapi/rights")
public class RightController {
    @Autowired
    private RightService rightService;
    @GetMapping
    public List<Right> getRightList(){
        return rightService.getRightList();
    }
    // adminapi/rights/3
    @PutMapping(value = "/{id}")
    public String updateRightList(@PathVariable Integer id,@RequestBody Right right){
        try {
            right.setId(id);
            rightService.updateRightlist(right);
            return "update-success";
        } catch (Exception e) {
            return "update-fail";
        }
    }
}
