package com.nj.back.controller;

import com.nj.back.config.OPStreamUDPClient;
import com.nj.back.config.UDPConfig;
import com.nj.back.pojo.Result;
import com.nj.back.pojo.SleepStage;
import com.nj.back.service.PyService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.io.*;
import java.net.DatagramPacket;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;


//1    python getEEG.py --file liangxiaotong_mcs_20181018_1_00.npz --select 0 --begin 0    select中0代表x，1代表y，每次默认返回500个值
//2    python readNPZ.py --file liangxiaotong_mcs_20181018_1_00.npz --select 0 --begin 0   select中0代表意识分类，1代表睡眠分期，意识分类只返回MCS或VS，睡眠分期每次每次默认返回500个值
//3    python prepare_sys.py --file data\sleepedf\sleep-cassette\patient_mcs_modehuai_20180928_1.edf  这个用于意识分类的预处理，会保存相应的npz
//4    python predict_sys.py --file data\sleepedf\sleep-cassette\eeg_eog\patient_mcs_modehuai_20180928_1.npz --config_file config/sleepedf.py --model_dir out_sleepedf/train --output_dir out_sleepedf/predict --log_file out_sleepedf/predict.log --use-best
//     这个命令是用于执行意识分类的代码的预测，对于单个文件进行预测，执行完后需要在执行命令2以获取对应的预测结果
//     上传文件，一般edf存放再指定的路径 resources/edf/
//     上传的npz文件，一般是要符合要求才能上传到睡眠分期和意识分类模型中的预处理npz的目录下

@RestController
@RequestMapping("/adminapi")
public class PyController {
    @Autowired
    private PyService pyService;
    @Autowired
    private OPStreamUDPClient opStreamUDPClient;

    private String post = "0";
    private int a1 = 0;
    private int j = 0;
    private int state = 0;

    private String getJarPath() throws IOException {
        List<String> processParameters = new ArrayList<String>();
        processParameters.add("cmd.exe");
        processParameters.add("/C");
        processParameters.add("echo");
        processParameters.add("%SLEEPNET_JAR_PATH%");
        ProcessBuilder pb = new ProcessBuilder(processParameters);
        Process p = pb.start();
        BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
        String line;
        StringBuilder output = new StringBuilder();
        while ((line = reader.readLine()) != null) {
            output.append(line);
        }
        return output.toString();
    }

    private String getEnvPy() throws IOException {
        List<String> processParameters = new ArrayList<String>();
        processParameters.add("cmd.exe");
        processParameters.add("/C");
        processParameters.add("echo");
        processParameters.add("%SLEEPNET%");
        ProcessBuilder pb = new ProcessBuilder(processParameters);
        Process p = pb.start();
        BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
        String line;
        StringBuilder output = new StringBuilder();
        while ((line = reader.readLine()) != null) {
            output.append(line);
        }
        return output.toString();
    }

    class MyThread extends Thread {
        private int counter;

        public MyThread(int counter) {
            this.counter = counter;
        }

        @Override
        public void run() {
            if (counter == 1) {
                System.out.println("线程1");
                String envpy = null;
                try {
                    envpy = getEnvPy();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                String  str=envpy+" createData.py"+" --switch 1"; //连接脑电帽
                System.out.println(str);
                pyService.pyNaoCnetAndStop(str);
            } else {
                System.out.println("线程2");
                try {
                    Thread.sleep(5000); // 等待150s
                    //脑电监测
                    //睡眠分期
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    } //Thread

    MyThread t1 = new MyThread(1);

    @PostMapping("/bao1")
    public Result bao1(@RequestBody Map<String,Object> body) throws IOException {
        opStreamUDPClient.send("start");
        state = 1;
        System.out.println("连接脑电帽");
        return new Result<>(0,null,"脑环已经连接");
    }

    @PostMapping("/bao2")
    public Result bao2(@RequestBody Map<String,Object> body) throws IOException {
        opStreamUDPClient.send("stop");
        state = 2;
        return new Result<>(0,null,"结束连接脑环");
    }

    @PostMapping("/bao3")
    public Result bao3(@RequestBody Map<String,Object> body) throws IOException {
        if (state == 0) {
            return new Result<>(406,0,"未连接");
        }
        else if(state == 1){
            return new Result<>(0,1,"已连接");
        }
        else if (state == 2){
            return new Result<>(0,2,"用户中断，断开连接");
        }
        return new Result<>(0,1,"连接成功");
    }

    @PostMapping("/naoReadStage")
    public Result naoReadStage(@RequestBody Map<String,Object> body) throws Exception {
        String begin = body.get("begin").toString();
        opStreamUDPClient.send("readStage_"+begin);
        System.out.println("发送读stage请求");
        List<SleepStage> sleepDataList = pyService.pyNaoStage(opStreamUDPClient.receive(new DatagramPacket(new byte[1377],1377)));
        if (sleepDataList.isEmpty()) {
            return new Result<>(0, null, "实时睡眠分期获取为null");
        }
        return new Result<>(1, sleepDataList, "实时睡眠分期获取成功");
    }
    @PostMapping("/naoGetEEG")
    public Result naoGetEEG(@RequestBody Map<String,Object> body) throws Exception {
        String select = body.get("select").toString();
        String begin = body.get("begin").toString();
        if(select.equals("0")){
            opStreamUDPClient.send("getEEG_"+begin);
        }else{
            opStreamUDPClient.send("getEOG_"+begin);
        }
        List<Double> list= pyService.pyNaoGetEEg(opStreamUDPClient.receive(new DatagramPacket(new byte[10240],10240)));
        System.out.println(list.size());
        if (list.size()==0){
            System.out.println("EEG获取为null");
            return new Result<>(0,null,"EEG获取为null");
        }
        return new Result<>(1,list,"EEG获取成功");
    }

    //prepare和predic整合在一起了，以下是分别对睡眠分期和意识分类功能的先prepare生成并保存对应的npz文件，然后再分别进行predict，
    // 最后读取预测结果还是要调用readNPZ
    @PostMapping("/predict")
    public Result predict(@RequestBody Map<String,Object> body) throws IOException {
        String filename = body.get("file").toString();
        String file = filename.split("\\.")[0];
        String envpy = getEnvPy();
        String jarPath = getJarPath();
        //意识分类
        String str1pre=envpy+" "+jarPath+"/DOC_valuation/prepare_sys.py --file "+jarPath+"/edf/"+file+".edf";
        String str1=envpy+" -W ignore "+jarPath+"/DOC_valuation/predict_sys.py --file "+jarPath+"/DOC_valuation/data/sleepedf/sleep-cassette/eeg_eog/"+file+".npz"+" --use-best";
        //睡眠分期
        String str2pre=envpy+" "+jarPath+"/sleepstage/prepare_sys.py --file "+jarPath+"/edf/"+ file+".edf";
        String str2=envpy+" -W ignore "+jarPath+"/sleepstage/predict_sys.py --file "+jarPath+"/sleepstage/data/sleepedf/sleep-cassette/eeg_eog/"+file+".npz"+" --use-best"; //睡眠分期
        String strCompute = envpy+" "+jarPath+"/DOC_valuation/computeStageOfNPZ.py --file "+jarPath+"/sleepstage/out_sleepedf/predict/pred_"+file+".npz";
        Boolean flag= pyService.predict(str1pre,str1,str2pre,str2,strCompute,filename);
        if (flag){
            return new Result<>(1,null,"预测成功");
        }
        return new Result<>(0,null,"预测失败");
    }

    @PostMapping("/getEEG")
    public Result getEEG(@RequestBody Map<String,Object> body) throws IOException {
        String file = body.get("file").toString();
        String select = body.get("select").toString();
        String begin = body.get("begin").toString();
        String envpy = getEnvPy();
        String str=envpy+" getEEG.py --file "+file+" --select "+select+" --begin "+begin;
        List<Double> list= pyService.pyGetEEg(str);
        if (list.size()==0){
            return new Result<>(0,null,"EEG获取为null");
        }
        return new Result<>(1,list,"EEG获取成功");
    }

    @PostMapping("/readNPZ")
    public Result readNPZ(@RequestBody Map<String,Object> body) throws IOException {//读取predict的npz
        String file = "pred_"+body.get("file").toString();
        String select = body.get("select").toString();
        String begin = body.get("begin").toString();
        String envpy = getEnvPy();
        String  str=envpy+" readNPZ.py --file "+file+" --select "+select+" --begin "+begin;
        List<Double> list= pyService.pyReadNpz(str);
        if (list.size()==0){
            return new Result<>(0,null,"读取为null");
        }
        return new Result<>(1,list,"读取成功");
    }
    @PostMapping("/readStage")
    public Result readStage(@RequestBody Map<String,Object> body) throws IOException {//读取predict的npz
        String file = "pred_"+body.get("file").toString();
        String begin = body.get("begin").toString();
        String envpy = getEnvPy();
        String  str=envpy+" stageOfNPZ.py --file "+file+" --begin "+begin;
        List<SleepStage> sleepDataList = pyService.pyStageOfNPZ(str);
        System.out.println(sleepDataList.size());
        if (sleepDataList.isEmpty()) {
            return new Result<>(0, null, "睡眠分期获取为null");
        }
        return new Result<>(1, sleepDataList, "睡眠分期获取成功");
    }

    @PostMapping("/predict2")
    public Result predict2(@RequestBody Map<String,Object> body) throws IOException {
        String filename = body.get("file").toString();
        String file = filename.split("\\.")[0];
        String envpy = getEnvPy();
        String jarPath = getJarPath();
        //意识分类
        String str1=envpy+" -W ignore "+jarPath+"/DOC_valuation/predict_sys.py --file "+jarPath+"/DOC_valuation/data/sleepedf/sleep-cassette/eeg_eog/"+file+".npz"+" --use-best";
        //睡眠分期
        String str2=envpy+" -W ignore "+jarPath+"/sleepstage/predict_sys.py --file "+jarPath+"/sleepstage/data/sleepedf/sleep-cassette/eeg_eog/"+file+".npz"+" --use-best";
        String strCompute = envpy+" "+jarPath+"/DOC_valuation/computeStageOfNPZ.py --file "+jarPath+"/sleepstage/out_sleepedf/predict/pred_"+file+".npz";
        Boolean flag= pyService.predict2(str1,str2,strCompute,filename);
        if (flag){
            return new Result<>(1,null,"预测成功");
        }
        return new Result<>(0,null,"预测失败");
    }
}
