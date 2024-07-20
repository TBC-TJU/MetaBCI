package com.nj.back.pojo;

import lombok.Data;

@Data
public class SleepStage {
    private String date;
    private String status;

    public SleepStage(String date, String status) {
        this.date = date;
        this.status = status;
    }

}
