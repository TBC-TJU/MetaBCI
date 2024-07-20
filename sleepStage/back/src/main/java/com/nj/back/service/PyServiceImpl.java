package com.nj.back.service;

import com.nj.back.dao.HistoryDao;
import com.nj.back.pojo.History;
import com.nj.back.pojo.SleepStage;
import com.nj.back.service.PyService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Service;

import java.io.*;
import java.nio.charset.Charset;
import java.util.*;

//执行python命令获取数据结果

@Service
public class PyServiceImpl implements PyService {
//     edf文件保存在以一个目录下，而由于npz是预处理后的，且睡眠分期和意识分类的预处理不同，所以分别保存在各自的模型下
//1    python getEEG.py --file liangxiaotong_mcs_20181018_1_00.npz --select 0 --begin 0  getEEG   select中0代表x，1代表y，每次默认返回500个值
//2    python readNPZ.py --file liangxiaotong_mcs_20181018_1_00.npz --select 0 --begin 0  getEEG  select中0代表意识分类，1代表睡眠分期，意识分类只返回MCS或VS，睡眠分期每次每次默认返回500个值
//3    python prepare_sys.py --file data\sleepedf\sleep-cassette\patient_mcs_modehuai_20180928_1.edf  这个用于意识分类的预处理，会保存相应的npz
//4    python predict_sys.py --file data\sleepedf\sleep-cassette\eeg_eog\patient_mcs_modehuai_20180928_1.npz --config_file config/sleepedf.py --model_dir out_sleepedf/train --output_dir out_sleepedf/predict --log_file out_sleepedf/predict.log --use-best
//     这个命令是用于执行意识分类的代码的预测，对于单个文件进行预测，执行完后需要在执行命令2以获取对应的预测结果
//     先prepare，再predict，再readNPZ，readNPZ才有打印预测的结果

    //    SLEEPNET_JAR_PATH
    //prepare和predic整合在一起了，以下是分别对睡眠分期和意识分类功能的先prepare生成并保存对应的npz文件，然后再分别进行predict，
    // 最后读取预测结果还是要调用readNPZ
    @Autowired
    private HistoryDao historyDao;


    //通过cmd命令行获取%SLEEPNET_JAR_PATH%环境变量的值
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

    public String executeCommand(String[] cmd) throws IOException, InterruptedException {
        ProcessBuilder pb = new ProcessBuilder();
        pb.command(cmd);
        Process p = pb.start();
//        System.out.println(p.waitFor());
        BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream(),Charset.forName("GBK")));
        String line;
        StringBuilder output = new StringBuilder();
        while ((line = reader.readLine()) != null) {
            output.append(line);
        }
        return output.toString();
    }

    public Boolean predict(String str1pre,String str1,String str2pre,String str2,String strCompute,String file1) {
        String[] cmd1pre=str1pre.split(" ");
        String[] cmd2pre=str2pre.split(" ");
        String[] cmd1=str1.split(" ");
        String[] cmd2=str2.split(" ");
        String[] cmd3=strCompute.split(" ");
        String result = "";
        String result2= "";
        try{
            String jarPath= getJarPath();
            //睡眠分期
            executeCommand(cmd2pre);
            executeCommand(cmd2);
            result = executeCommand(cmd3);
            result=result.replaceAll("\\s+","");
            //意识分类
            executeCommand(cmd1pre);
            result2 = executeCommand(cmd1);
        }catch (IOException e){
            System.out.println("调用python脚本并读取结果时出错：" + e.getMessage()+" "+result);
            return false;
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        String[] strs=result.split(",");//逗号分割，上面的result应该是[ 1,2,3,4,5]，处理过后应该是1,2,3,4,5（返回这个）
        List<String> list = new ArrayList<>(Arrays.asList(strs)); //[2021-06-16 19:20:18,4.91h,4.91h,3.61h,1.30h,0.00h,0.00h,0.00%]
        String str = (result2.equals("1"))?"MCS":"VS";
        list.add(str);
        list.add(file1);
        //把list存入数据库
        int num = historyDao.countByFilename(list.get(14));
        list.add(String.valueOf(num));
        System.out.println("当前的id："+num);
        System.out.println(list); //字符串数组 list 示例：[2021-06-1619:20:18, 4.91h, 4.91h, 4.03h, 0.87h, 0.00h, 0.00h, 0.00h, 82.17%, 17.83%, 0.00%, 0.00%, 0.00%, VS]
        int count = historyDao.countByDate(list.get(0));
        if (count == 0) {
            // 如果不存在相同的 date 值，执行插入操作
            historyDao.insertHistory(new History(list));
            historyDao.updateFileTag(file1);
        } else {
            // 如果存在相同的 date 值，可以选择不做任何处理或者记录日志等
            // 可以选择不做任何处理以保持数据状态不变
        }

        //这里需要再加一下，修改为已测试
        return true;
    }

    public List<Double> pyReadNpz(String str) { //读取predict的npz
        String[] cmd=str.split(" ");
        String result = "";
        String select=cmd[5];
        try{
            String jarPath= getJarPath();
            cmd[1]=jarPath+"/DOC_valuation/"+cmd[1];
            if (Objects.equals(select, "0")){ //意识分类
                cmd[3]=jarPath+"/DOC_valuation/out_sleepedf/predict/"+cmd[3];
            }else if (Objects.equals(select, "1")){ //睡眠分期
                cmd[3]=jarPath+"/sleepstage/out_sleepedf/predict/"+cmd[3];
            }
            result=executeCommand(cmd);
            result=result.replaceAll("\\s+","");
        }catch (IOException e){
            System.out.println("调用python脚本并读取结果时出错：" + e.getMessage()+" "+result);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        String[] strs=result.split(",");//逗号分割，上面的result应该是[ 1,2,3,4,5]，处理过后应该是1,2,3,4,5（返回这个）
        List<Double> list=new ArrayList<>();
        for (String i:strs) {
            list.add(Double.parseDouble(i));
        }
        return list;
    }

    //获取python的输出，并转成double数组
    public List<Double> pyGetEEg(String str)   {
        String[] cmd=str.split(" ");
        String result = "";
        String file_type=cmd[3].split("\\.")[1];
        try{
            String jarPath= getJarPath();
            cmd[1]=jarPath+"/DOC_valuation/"+cmd[1];
            if (Objects.equals(file_type, "edf")){
                cmd[3]=jarPath+"/edf/"+cmd[3];
            }else if (Objects.equals(file_type, "npz")){ //默认为睡眠分期的npz
                cmd[3]= jarPath+"/sleepstage/data/sleepedf/sleep-cassette/eeg_eog/"+cmd[3];
            }else if (Objects.equals(file_type, "xls")){ //xls,不能为xlsx
                cmd[3]= jarPath+"/excel/"+cmd[3];
            }else if (Objects.equals(file_type, "csv")){ //csv
                cmd[3]= jarPath+"/csv/"+cmd[3];
            }
            result=executeCommand(cmd);
            result=result.replaceAll("\\s+","");
        }catch (IOException e){
            System.out.println("调用python脚本并读取结果时出错：" + e.getMessage()+" "+result);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        String[] strs=result.split(",");//逗号分割，上面的result应该是[ 1,2,3,4,5]，处理过后应该是1,2,3,4,5（返回这个）
        List<Double> list=new ArrayList<>();
        for (String i:strs) {
            list.add(Double.parseDouble(i));
        }
        return list;
    }

    public List<SleepStage> pyStageOfNPZ(String str) {
        String[] cmd=str.split(" ");
        String result = "";
        try{
            String jarPath= getJarPath();
            cmd[1]=jarPath+"/DOC_valuation/"+cmd[1];
            cmd[3]=jarPath+"/sleepstage/out_sleepedf/predict/"+cmd[3];
            result=executeCommand(cmd);
        }catch (IOException e){
            System.out.println("调用python脚本并读取结果时出错：" + e.getMessage()+" "+result);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        String[] strs=result.split(",");//逗号分割，上面的result应该是[ 1,2,3,4,5]，处理过后应该是1,2,3,4,5（返回这个）
        List<SleepStage> sleepDataList = new ArrayList<>();
        for (String i : strs) {
            String[] temp = i.split(" ");
            String date = temp[0] + " " + temp[1];
            String status = temp[2];
            SleepStage sleepStage = new SleepStage(date, status);
            sleepDataList.add(sleepStage);
        }
        return sleepDataList;
    }


    public Boolean predict2(String str1,String str2,String strCompute,String file1) {
        String[] cmd1=str1.split(" ");
        String[] cmd2=str2.split(" ");
        String[] cmd3=strCompute.split(" ");
        String result = "";
        String result2= "";
        try{
            String jarPath= getJarPath();
            //睡眠分期 预测
            result2 = executeCommand(cmd2);
            //意识分类 预测
            executeCommand(cmd1);
            result = executeCommand(cmd3);
            result=result.replaceAll("\\s+","");
        }catch (IOException e){
            System.out.println("调用python脚本并读取结果时出错：" + e.getMessage()+" "+result);
            return false;
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        String[] strs=result.split(",");//逗号分割，上面的result应该是[ 1,2,3,4,5]，处理过后应该是1,2,3,4,5（返回这个）
        List<String> list = new ArrayList<>(Arrays.asList(strs)); //[2021-06-16 19:20:18,4.91h,4.91h,3.61h,1.30h,0.00h,0.00h,0.00%]
        String str = (result2.equals("1"))?"MCS":"VS";
        list.add(str);
        list.add(file1);
        //把list存入数据库
        int num = historyDao.countByFilename(list.get(14));
        list.add(String.valueOf(num));
        System.out.println("当前的id："+num);
        System.out.println(list); //字符串数组 list 示例：[2021-06-1619:20:18, 4.91h, 4.91h, 4.03h, 0.87h, 0.00h, 0.00h, 0.00h, 82.17%, 17.83%, 0.00%, 0.00%, 0.00%, VS]
        int count = historyDao.countByDate(list.get(0));
        if (count == 0) {
            // 如果不存在相同的 date 值，执行插入操作
            historyDao.insertHistory(new History(list));
            historyDao.updateFileTag(file1);
        } else {
            // 如果存在相同的 date 值，可以选择不做任何处理或者记录日志等
            // 可以选择不做任何处理以保持数据状态不变
        }
        //这里需要再加一下，修改为已测试
        return true;
    }

    public List<SleepStage> pyNaoStage(String result) {
        System.out.println(result);
        String[] strs=result.split(",");//逗号分割，上面的result应该是[ 1,2,3,4,5]，处理过后应该是1,2,3,4,5（返回这个）
        List<SleepStage> sleepDataList = new ArrayList<>();
        for (String i : strs) {
            String[] temp = i.split(" ");
            String date = temp[0] + " " + temp[1];
            String status = temp[2];
            SleepStage sleepStage = new SleepStage(date, status);
            sleepDataList.add(sleepStage);
        }
        return sleepDataList;
    }
    public List<Double> pyNaoGetEEg(String result)   {
        String[] strs=result.split(",");//逗号分割，上面的result应该是[ 1,2,3,4,5]，处理过后应该是1,2,3,4,5（返回这个）
        List<Double> list=new ArrayList<>();
        for (String i:strs) {
            list.add(Double.parseDouble(i));
        }
        return list;
    }

    public Integer pyNaoIsPrepare(String str) {
        String[] cmd=str.split(" ");
        String result = "";
        try{
            String jarPath= getJarPath();
            cmd[1]=jarPath+"/sleepstage/"+cmd[1];
            result=executeCommand(cmd);
        }catch (IOException e){
            System.out.println("调用python脚本并读取结果时出错：" + e.getMessage()+" "+result);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        System.out.println(result);
        return Integer.parseInt(result);
    }

    public void pyNaoCnetAndStop(String str) {
        String[] cmd=str.split(" ");
        String result = "";
        try{
            String jarPath= getJarPath();
            cmd[1]=jarPath+"/sleepstage/"+cmd[1];
            result=executeCommand(cmd);
        }catch (IOException e){
            System.out.println("调用python脚本并读取结果时出错：" + e.getMessage()+" "+result);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

}
