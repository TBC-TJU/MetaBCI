package com.nj.back.controller;

import com.nj.back.service.UploadService;
import com.nj.back.utils.ResultOBJ;
import com.nj.back.utils.SYSConstant;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.FileNotFoundException;
import java.io.IOException;

@RequestMapping("/adminapi/upload")
@RestController
public class UploadController {

    @Autowired
    private UploadService uploadService;

    @PostMapping
    public ResultOBJ upload(@RequestParam("file") MultipartFile file){
        try{
            uploadService.upload(file);
            return new ResultOBJ(SYSConstant.CODE_SUCCESS,"上传成功");
        }catch (IOException e){
            throw new RuntimeException(e);
        }

    }
}
