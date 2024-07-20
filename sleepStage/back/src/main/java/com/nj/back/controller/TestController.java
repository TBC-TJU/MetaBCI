package com.nj.back.controller;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.util.ClassUtils;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.util.Objects;
import java.util.UUID;

@RestController
public class TestController {
    @PostMapping("adminapi/testupload")
    public String Test18(HttpServletRequest request, @RequestParam("file") MultipartFile file) throws UnsupportedEncodingException {
        String filename = file.getOriginalFilename();
        assert filename != null;
        String filetype = filename.substring(filename.lastIndexOf("."));
        String newfile = UUID.randomUUID()+filetype;
        String classespath = Objects.requireNonNull(Objects.requireNonNull(ClassUtils.getDefaultClassLoader()).getResource("")).getPath();
        /*解决文件路径中的空格问题*/
        String decodeClassespath = URLDecoder.decode(classespath,"utf-8");
        System.out.println(decodeClassespath);
        /**/
        File file1 = new File("D:/dcsystem/file/"+newfile);
        if(!file1.exists()){
            file1.mkdirs();
        }
        try {
            file.transferTo(file1);
            return "上传成功";
        } catch (IOException e) {
            e.printStackTrace();
        }
        return "上传失败";
    }
}
