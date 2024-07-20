package com.nj.back.pojo;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

@Data
@NoArgsConstructor
public class Result<T> implements Serializable {
    private Object data;
    private Integer code;
    private String msg;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Integer total;

    @JsonIgnore
    private Map<String,Object> map = new HashMap<String,Object>();


    public Result(Integer code, Object data) {
        this.data = data;
        this.code = code;
    }

    public Result(Integer code, Object data, String msg) {
        this.data = data;
        this.code = code;
        this.msg = msg;
    }
    public Result(Integer code, Object data, String msg, Integer total) {
        this.data = data;
        this.code = code;
        this.msg = msg;
        this.total=total;
    }

    public Result<T> data(Map<String,Object> map){
        this.setData(map);
        return this;
    }
}
