package com.nj.back.service;

import org.springframework.web.multipart.MultipartFile;

import java.io.FileNotFoundException;
import java.io.IOException;

public interface UploadService {
    public void upload(MultipartFile file) throws IOException;
}
