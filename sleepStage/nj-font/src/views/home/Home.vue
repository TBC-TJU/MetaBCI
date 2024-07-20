<template class="bg">
  <div class="bg" style="padding-top: 10px">
    <div class="bk">
      <div class="content">
        <div
          class="ti"
          style="
            position: fixed;
            top: 10px;
            margin-left: -50px;
            color: white;
            font-size: 30px;
          "
        >
          首页
        </div>
        <div class="t">
          NeuroDoc<br />意识障碍睡眠分期和评估 <br />
          <div
            style="
              font-size: 15px;
              float: right;
              width: 400px;
              margin-top: 10px;
              margin-left: 200px;
              justify-content: initial;
              font-weight: normal;
            "
          >
            ——针对意识障碍患者，提供睡眠的分期检测和评估报告
          </div>
        </div>
        <!--简介-->
        <div class="container">
          <div>
            <img src="public\mind2.png" class="image right" />
          </div>
          <div><img src="public\doctor.png" class="image top" /></div>
          <div><img src="public\computer.png" class="image bottom" /></div>

          <div><img src="public\mind.png" class="image left" /></div>
        </div>
      </div>
    </div>
    <div class="bk1">
      <div style="float: left; width: 600px; margin: 40px; line-height: 40px">
        <p style="font-weight: bold; font-size: 20px">项目背景：</p>
        &nbsp;&nbsp;&nbsp;&nbsp;据不完全统计，我国现有至少50万意识障碍患者，而且每年新增人数超过10万，年累计医疗开支达300到500亿元。大量的意识障碍患者给家庭和社会带来了巨大的痛苦和经济压力，也给医疗系统造成了极大负担。
        <br />&nbsp;&nbsp;&nbsp;&nbsp;目前睡眠分期的工作仍然是临床专家手动进行，耗时耗力，效率较低。每一次8小时的睡眠，每30秒作为一个睡眠时期，需要划分960个阶段，工作量巨大。且手工的睡眠分期具有较强的主观性，要求医生具有丰富的临床经验。因此需要研究高效的自动睡眠分期方法来帮助医生进行睡眠分期。
      </div>
      <div><img src="public\doctor.png" class="doc" /></div>
    </div>
    <div class="bk1">
      <div><img src="public\mind.png" class="mind" /></div>
      <div
        style="
          float: left;
          width: 600px;
          margin: 40px;
          margin-left: 120px;
          line-height: 40px;
        "
      >
        <p style="font-weight: bold; font-size: 20px">项目背景：</p>
        &nbsp;&nbsp;&nbsp;&nbsp;研究发现即使引入目前标准化行为学评估工具CRS-R量表，意识障碍仍存在40%左右的误诊率。而目前国内大多数地区，仍处在标准化行为评估尚未开展或存在操作不规范等情况，因此我国目前意识障碍的误诊率应远高于40%。行为量表是一种基于任务态的诊断评估方法。该方法需要患者对指令做出反应，导致误诊率高。而通过静息态睡眠无需患者进行任何动作，只需采集患者睡眠时的脑电数据进行分析处理，可以大大降低误诊率。
      </div>
    </div>
    <div class="bk1">
      <div style="float: left; width: 600px; margin: 30px; line-height: 40px">
        <p style="font-weight: bold; font-size: 20px">社会价值:</p>
        &nbsp;&nbsp;&nbsp;&nbsp;当下国内多数医疗机构对意识障碍诊断不明，盲目促醒。通过本系统的评估，医生可以更加针对性地分类管理和治疗，减少国家医保基金及家庭医疗支出，减少因病致贫。
        <br />&nbsp;&nbsp;&nbsp;&nbsp;患者家属缺乏心理疏导，心理压力大。通过系统的评估家属心里才能得到安宁。患者家庭不仅需要支付巨额的医疗费和给予患者长期陪伴，还要遭受社会舆论的影响，在心理与生理的双重强压下，可能出现严重的心理疾病。不少家属不愿放弃，投入了大量金钱与时间去追求微乎其微的可能性。这不仅造成了资源的浪费，对患者本人而言也得不到合理妥善的安置方案。
      </div>
      <div><img src="public\mind2.png" class="mind2" /></div>
    </div>
    <div class="bk1">
      <div><img src="public\computer.png" class="mind" /></div>
      <div
        style="
          float: left;
          width: 600px;
          margin: 40px;
          margin-left: 80px;
          line-height: 40px;
        "
      >
        <p style="font-weight: bold; font-size: 20px">社会价值：</p>
        &nbsp;&nbsp;&nbsp;&nbsp; 推动社会公益事业、推动相关政策改革、探索医养康护新模式。
      </div>
    </div>
  </div>
</template>

<style>
.bk {
  height: 500px;
  width: 1200px;
  background-color: white;
  margin-left: 60px;
  margin-bottom: 20px;
  top: 10px;
  border-radius: 50px;
}
.bk1 {
  height: 400px;
  width: 1200px;
  background-color: white;
  margin-left: 60px;
  margin-bottom: 20px;
  border-radius: 50px;
}

.bg {
  background-color: #f6f6f6;
  margin: 0px;
  padding: 0px;
}

.content {
  display: flex;
}

.t {
  font-size: 46px;
  font-weight: bold;
  width: 600px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  margin-top: 120px;
}

.container {
  width: 200px;
  height: 200px;
  position: relative;
  display: flex;
}

.image {
  position: absolute;
  width: 100px;
  height: 100px;
  background-color: #3498db;
}

.top {
  top: 10px;
  left: 100px;
  width: 250px;
  height: 166px;
  border-radius: 20px;
}

.bottom {
  top: 250px;
  left: 100px;
  width: 209px;
  height: 194px;
  border-radius: 20px;
}

.left {
  top: 150px;
  width: 181px;
  height: 148px;
  left: 0px;
  border-radius: 20px;
}

.right {
  top: 150px;
  left: 250px;
  width: 286px;
  height: 136px;
  border-radius: 20px;
}
.doc {
  width: 483px;
  height: 321px;
  border-radius: 20px;
  float: right;
  margin-top: 20px;
  margin-right: 20px;
}
.mind {
  width: 292px;
  height: 239px;
  border-radius: 20px;
  float: left;
  margin-top: 60px;
  margin-left: 20px;
}
.mind2 {
  width: 469px;
  height: 223px;
  border-radius: 20px;
  float: right;
  margin-top: 20px;
  margin-right: 20px;
}
.computer {
  top: 150px;
  left: 250px;
  width: 285px;
  height: 264px;
  border-radius: 20px;
}
</style>
