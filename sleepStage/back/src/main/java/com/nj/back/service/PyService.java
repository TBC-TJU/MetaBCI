package com.nj.back.service;

import com.nj.back.pojo.SleepStage;

import java.util.List;

//     edf文件保存在以一个目录下，而由于npz是预处理后的，且睡眠分期和意识分类的预处理不同，所以分别保存在各自的模型下
//1    python getEEG.py --file liangxiaotong_mcs_20181018_1_00.npz --select 0 --begin 0  getEEG   select中0代表x，1代表y，每次默认返回500个值
//2    python readNPZ.py --file liangxiaotong_mcs_20181018_1_00.npz --select 0 --begin 0  getEEG  select中0代表意识分类，1代表睡眠分期，意识分类只返回MCS或VS，睡眠分期每次每次默认返回500个值
//3    python prepare_sys.py --file data\sleepedf\sleep-cassette\patient_mcs_modehuai_20180928_1.edf  这个用于意识分类的预处理，会保存相应的npz
//4    python predict_sys.py --file data\sleepedf\sleep-cassette\eeg_eog\patient_mcs_modehuai_20180928_1.npz --config_file config/sleepedf.py --model_dir out_sleepedf/train --output_dir out_sleepedf/predict --log_file out_sleepedf/predict.log --use-best
//     这个命令是用于执行意识分类的代码的预测，对于单个文件进行预测，执行完后需要在执行命令2以获取对应的预测结果
//     先prepare，再predict，再readNPZ，readNPZ才有打印预测的结果

//prepare和predic整合在一起了，以下是分别对睡眠分期和意识分类功能的先prepare生成并保存对应的npz文件，然后再分别进行predict，
// 最后读取预测结果还是要调用readNPZ

public interface PyService {
    List<Double> pyGetEEg(String str);
    List<Double> pyReadNpz(String str);
    List<SleepStage> pyStageOfNPZ(String str);
    Boolean predict(String str1pre,String str1,String str2pre,String str2,String strCompute,String filename);

    Boolean predict2(String str1, String str2, String strCompute, String filename);
    List<SleepStage> pyNaoStage(String str);

    List<Double> pyNaoGetEEg(String str);
    Integer pyNaoIsPrepare(String str);
    void pyNaoCnetAndStop(String str);
}
