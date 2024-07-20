package com.nj.back.pojo;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.NoArgsConstructor;

import java.util.List;

@TableName(value = "history")
@NoArgsConstructor
public class History {
    private int id;
    private String date;
    private String time;
    private String sleeptime;
    private String wtime;
    private String n1time;
    private String n2time;
    private String n3time;
    private String remtime;
    private String wrate;
    private String n1rate;
    private String n2rate;
    private String n3rate;
    private String remrate;
    private String type;
    private String upload;

    public String getUpload() {
        return upload;
    }

    public void setUpload(String upload) {
        this.upload = upload;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public String getTime() {
        return time;
    }

    public void setTime(String time) {
        this.time = time;
    }

    public String getSleeptime() {
        return sleeptime;
    }

    public void setSleeptime(String sleeptime) {
        this.sleeptime = sleeptime;
    }

    public String getWtime() {
        return wtime;
    }

    public void setWtime(String wtime) {
        this.wtime = wtime;
    }

    public String getN1time() {
        return n1time;
    }

    public void setN1time(String n1time) {
        this.n1time = n1time;
    }

    public String getN2time() {
        return n2time;
    }

    public void setN2time(String n2time) {
        this.n2time = n2time;
    }

    public String getN3time() {
        return n3time;
    }

    public void setN3time(String n3time) {
        this.n3time = n3time;
    }

    public String getRemtime() {
        return remtime;
    }

    public void setRemtime(String remtime) {
        this.remtime = remtime;
    }

    public String getWrate() {
        return wrate;
    }

    public void setWrate(String wrate) {
        this.wrate = wrate;
    }

    public String getN1rate() {
        return n1rate;
    }

    public void setN1rate(String n1rate) {
        this.n1rate = n1rate;
    }

    public String getN2rate() {
        return n2rate;
    }

    public void setN2rate(String n2rate) {
        this.n2rate = n2rate;
    }

    public String getN3rate() {
        return n3rate;
    }

    public void setN3rate(String n3rate) {
        this.n3rate = n3rate;
    }

    public String getRemrate() {
        return remrate;
    }

    public void setRemrate(String remrate) {
        this.remrate = remrate;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public History(int id, String date, String time, String sleeptime, String wtime, String n1time, String n2time, String n3time, String remtime, String wrate, String n1rate, String n2rate, String n3rate, String remrate, String type) {
        this.id = id;
        this.date = date;
        this.time = time;
        this.sleeptime = sleeptime;
        this.wtime = wtime;
        this.n1time = n1time;
        this.n2time = n2time;
        this.n3time = n3time;
        this.remtime = remtime;
        this.wrate = wrate;
        this.n1rate = n1rate;
        this.n2rate = n2rate;
        this.n3rate = n3rate;
        this.remrate = remrate;
        this.type = type;
    }

    public History(String date, String time, String sleeptime, String wtime, String n1time, String n2time, String n3time, String remtime, String wrate, String n1rate, String n2rate, String n3rate, String remrate, String type) {
        this.date = date;
        this.time = time;
        this.sleeptime = sleeptime;
        this.wtime = wtime;
        this.n1time = n1time;
        this.n2time = n2time;
        this.n3time = n3time;
        this.remtime = remtime;
        this.wrate = wrate;
        this.n1rate = n1rate;
        this.n2rate = n2rate;
        this.n3rate = n3rate;
        this.remrate = remrate;
        this.type = type;
    }

    public History(List<String> list) {
        this.date = list.get(0);
        this.time = list.get(1);
        this.sleeptime = list.get(2);
        this.wtime = list.get(3);
        this.n1time = list.get(4);
        this.n2time = list.get(5);
        this.n3time = list.get(6);
        this.remtime = list.get(7);
        this.wrate = list.get(8);
        this.n1rate = list.get(9);
        this.n2rate = list.get(10);
        this.n3rate = list.get(11);
        this.remrate = list.get(12);
        this.type = list.get(13);
        this.upload =list.get(14);
        this.id = Integer.parseInt(list.get(15));
    }
}
