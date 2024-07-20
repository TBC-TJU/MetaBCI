package com.nj.back.pojo;

import lombok.Data;

@Data
public class User {
    private Integer id;
    private String name;
    private String tag;
    private String date;
    private String type;
    private String content;
    private String upload;
    private String agree;
}
