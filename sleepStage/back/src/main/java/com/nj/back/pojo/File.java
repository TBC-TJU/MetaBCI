package com.nj.back.pojo;

import lombok.Data;

@Data
public class File {
    private Integer id;
    private String name;
    private String type;
    private String content;
    private String upload;
    private String agree;
    private String tag;
    private String date;
}
