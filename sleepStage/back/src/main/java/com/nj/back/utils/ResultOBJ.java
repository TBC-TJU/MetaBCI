package com.nj.back.utils;

import lombok.Data;

@Data
public class ResultOBJ {
    private Integer code;
    private  String msg;
    private Object data;

    public ResultOBJ(Integer code){
        this.code =code;
    }

    public ResultOBJ(Integer code,String msg){
        this.code = code;
        this.msg= msg;
    }
    public ResultOBJ(Integer code,String msg,Object data){
        this.code = code;
        this.msg = msg;
        this.data = data;
    }

    public  static  final  ResultOBJ ADD_SUCCESS = new ResultOBJ(SYSConstant.CODE_SUCCESS,SYSConstant.ADD_SUCCESS);

    public  static  final  ResultOBJ ADD_ERROR = new ResultOBJ(SYSConstant.CODE_ERROR,SYSConstant.ADD_ERROR);

    public  static  final  ResultOBJ UPDATE_SUCCESS = new ResultOBJ(SYSConstant.CODE_SUCCESS,SYSConstant.UPDATE_SUCCESS);

    public  static  final  ResultOBJ UPDATE_ERROR = new ResultOBJ(SYSConstant.CODE_ERROR,SYSConstant.UPDATE_ERROR);

    public  static  final  ResultOBJ DELETE_SUCCESS = new ResultOBJ(SYSConstant.CODE_SUCCESS,SYSConstant.DELETE_SUCCESS);

    public  static  final  ResultOBJ DELETE_ERROR = new ResultOBJ(SYSConstant.CODE_ERROR,SYSConstant.DELETE_ERROR);

}
