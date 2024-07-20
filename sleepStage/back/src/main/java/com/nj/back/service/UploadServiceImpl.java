package com.nj.back.service;

import org.springframework.stereotype.Service;
import org.springframework.util.ResourceUtils;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.IIOException;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.ResourceBundle;

//上传文件


@Service
public class UploadServiceImpl implements UploadService {
    @Override
    public void upload(MultipartFile file) throws IOException {
        List<String> processParameters = new ArrayList<String>();
        processParameters.add("cmd.exe");
        processParameters.add("/C");
        processParameters.add("echo");
        processParameters.add("%SLEEPNET_JAR_PATH%");
        ProcessBuilder pb0 = new ProcessBuilder(processParameters);
        Process p0 = pb0.start();
        BufferedReader reader0 = new BufferedReader(new InputStreamReader(p0.getInputStream()));
        String line0;
        StringBuilder output0 = new StringBuilder();
        while ((line0 = reader0.readLine()) != null) {
            output0.append(line0);
        }
        String jarPath = output0.toString();

        String originalFilename = file.getOriginalFilename();
        String type;
        if (originalFilename != null) {
            type = originalFilename.split("\\.")[1];
            if (Objects.equals(type, "xlsx")) type="xls";
            String filepath,filepath2;
            switch (type) {
                case "edf":
                    // 获取文件保存目录的路径，这里使用classpath:static/upload/
                    filepath = jarPath + "/edf/";
                    System.out.printf(filepath);
                    System.out.printf(jarPath);
                    // 使用原始文件名作为保存的文件名
                    File dest = new File(filepath + originalFilename);
                    if (!dest.exists()) {
                        dest.mkdirs();
                    }
                    // 将上传文件保存到目标文件
                    try {
                        file.transferTo(dest);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    break;
                case "npz":
                    // 获取文件保存目录的路径，这里使用classpath:static/upload/
                    filepath = jarPath + "/MCASleepNet/data/sleepedf/eeg_eog/";
                    filepath2 = jarPath + "/MCASleepNet_sys/data/sleepedf/eeg_eog/";
                    System.out.println(filepath);
                    System.out.println(filepath2);
                    System.out.println(jarPath);
                    // 使用原始文件名作为保存的文件名
                    File dest2 = new File(filepath + originalFilename);
                    File dest3 = new File(filepath2 + originalFilename);
                    if (!dest2.exists()) {
                        dest2.mkdirs();
                    }
                    if (!dest3.exists()) {
                        dest3.mkdirs();
                    }
                    // 将上传文件保存到目标文件
                    try {
                        file.transferTo(dest2);
                        file.transferTo(dest3);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                case "xls":
                    // 获取文件保存目录的路径，这里使用classpath:static/upload/
                    filepath = jarPath + "/excel/";
                    System.out.printf(filepath);
                    System.out.printf(jarPath);
                    // 使用原始文件名作为保存的文件名
                    File dest4 = new File(filepath + originalFilename);
                    if (!dest4.exists()) {
                        dest4.mkdirs();
                    }
                    // 将上传文件保存到目标文件
                    try {
                        file.transferTo(dest4);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    break;
                case "csv":
                    // 获取文件保存目录的路径，这里使用classpath:static/upload/
                    filepath = jarPath + "/csv/";
                    System.out.printf(filepath);
                    System.out.printf(jarPath);
                    // 使用原始文件名作为保存的文件名
                    File dest5 = new File(filepath + originalFilename);
                    if (!dest5.exists()) {
                        dest5.mkdirs();
                    }
                    // 将上传文件保存到目标文件
                    try {
                        file.transferTo(dest5);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                    break;
            }
        }
    }
}
