package com.nj.back.service;

import com.nj.back.dao.FileMapper;
import com.nj.back.pojo.File;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Service;

@Service
public class FileServicelmpl implements FileService{
    @Autowired
    private FileMapper fileMapper;

    @Override
    public void updateFile(File file) {
        try {
            fileMapper.updateFile(file);
        }catch ( DuplicateKeyException e){

        }

    }
}
