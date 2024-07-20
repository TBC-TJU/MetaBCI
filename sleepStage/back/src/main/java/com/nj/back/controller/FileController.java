package com.nj.back.controller;

import com.nj.back.pojo.File;
import com.nj.back.service.FileService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;


@RestController
@RequestMapping("/adminapi")
public class FileController {
    @Autowired
    private FileService fileService;

    @PutMapping("/file")
    public String updateFile(@RequestBody File file){
        try {
            fileService.updateFile(file);
            return "update-success";
        } catch (Exception e) {
            return "update-fail";
        }
    }
}
