<template>
  <div class="bg">
    <div
      class="ti"
      style="position: fixed; top: 10px; margin-left: 10px; color: white; font-size: 30px"
    >
      脑电监测
    </div>
    <div class="common-layout">
      <el-container>
        <el-main style="margin-top: 0px; padding-top: 0px">
          <el-scrollbar style="height: 100%; margin-top: 0px">
            <el-container>
              <!--EEG，EOG，BCI板块-->
              <el-header style="height: 200px">
                <div style="display: flex; border-radius: 10px">
                  <div class="grid-content ep-bg-purple">
                    <div style="display: flex">
                      <div style="flex: 1">
                        <div
                          style="
                            font-size: 40px;
                            font-weight: bold;
                            margin: 20px;
                            margin-bottom: 10px;
                            margin-left: 10px;
                          "
                        >
                          EEG
                        </div>
                        <div
                          style="
                            font-size: 18px;
                            margin: 10px;
                            color: black;
                            width: 160px;
                          "
                        >
                          脑电信号，<br />Electroencephalogram
                        </div>
                      </div>
                      <div style="flex: 1">
                        <img src="public\line.png" style="width: 180px" />
                      </div>
                    </div>
                  </div>
                  <div class="grid-content ep-bg-purple">
                    <div style="display: flex">
                      <div style="flex: 1">
                        <div
                          style="
                            font-size: 40px;
                            font-weight: bold;
                            margin: 20px;
                            margin-bottom: 10px;
                            margin-left: 10px;
                          "
                        >
                          EOG
                        </div>
                        <div
                          style="
                            font-size: 18px;
                            margin: 10px;
                            color: black;
                            width: 160px;
                          "
                        >
                          眼电信号，<br />Electro-oculogram
                        </div>
                      </div>
                      <div style="flex: 1">
                        <img src="public\eye.png" style="width: 100px; margin: 30px" />
                      </div>
                    </div>
                  </div>
                  <div class="grid-content ep-bg-purple">
                    <div
                      style="
                        display: flex;
                        background-color: #0c4f9c;
                        border-radius: 10px;
                        height: 170px;
                      "
                    >
                      <div style="flex: 1">
                        <div
                          style="
                            font-size: 40px;
                            font-weight: bold;
                            margin: 20px;
                            margin-bottom: 10px;
                            margin-left: 10px;
                            color: white;
                          "
                        >
                          BCI
                        </div>
                        <div
                          style="
                            font-size: 18px;
                            margin: 10px;
                            color: white;
                            width: 160px;
                          "
                        >
                          脑机接口，<br />Brain-Computer Interface
                        </div>
                      </div>
                      <div style="flex: 1">
                        <img src="public\jiekou.png" style="width: 130px; margin: 20px" />
                      </div>
                    </div>
                  </div>
                </div>
              </el-header>
              <!--脑电数值图-->
              <el-main>
                <div>
                  <el-container
                    class="all"
                    style="margin-bottom: 30px; margin-top: 0px; border-radius: 10px"
                  >
                    <el-header
                      class="title"
                      style="height: 40px; width: auto; border-radius: 10px"
                      >脑电数值图</el-header
                    >
                    <el-main
                      style="
                        height: 350px;
                        overflow-y: auto;
                        background-color: white;
                        border-radius: 10px;
                      "
                    >
                      <e-charts id="naodianshuju" class="chart" :option="option1" />
                      <!-- <echart2></echart2> -->
                    </el-main>
                  </el-container>
                </div>
              </el-main>
              <!--其他板块-->
              <el-footer>
                <div style="display: flex; border-radius: 50px; margin-top: -20px">
                  <!--选择被试-->
                  <div style="display: flex; flex: 1; height: 400px">
                    <el-container class="all" style="width: 300px">
                      <el-header class="title" style="height: 40px">选择被试</el-header>
                      <el-main style="background-color: white">
                        <div>
                          <el-table
                            ref="multipleTableRef"
                            :data="tableData"
                            style="height: 310px; top: -10px; border-radius: 10px"
                            @select="selectClick"
                            stripe
                            class="custom-table test"
                            @selection-change="handleSelectionChange"
                            :header-cell-style="{
                              background: '#0c4f9c',
                              color: 'white',
                            }"
                          >
                            <el-table-column type="selection" width="55" />
                            <el-table-column label="上传日期" width="120">
                              <template #default="scope">{{ scope.row.date }}</template>
                            </el-table-column>

                            <el-table-column property="name" label="名称" width="auto" />
                          </el-table>
                        </div>
                      </el-main>
                    </el-container>
                  </div>
                  <!--开始分析-->
                  <div style="flex: 1">
                    <div
                      style="
                        flex: 1;
                        height: 300px;
                        background-color: #518aca;
                        margin-left: 20px;
                        margin-right: 20px;
                        border-radius: 10px;
                      "
                    >
                      <div style="width: 200px; margin-left: 25%; margin-bottom: -20px">
                        <img src="/public/logo.png" style="width: 150px" />
                      </div>
                      <div style="display: flex">
                        <!--
                        <el-button type="success"  style="margin-left: 10px;width: 50px;height: 50px;borader: radius 20px;background-color:white;color:rgb(20, 122, 88);font-size:20px;flex:1;" @click="startAnalysis1">链接脑环</el-button>
                      -->

                        <el-button
                          type="success"
                          style="
                            margin-left: 20%;
                            margin-right: 10px;
                            width: 50px;
                            height: 60px;
                            borader: 50px solid gray;

                            background-color: transparent;
                            font-size: 25px;
                            flex: 0.8;
                          "
                          @click="startAnalysis"
                          >开始分析</el-button
                        >
                      </div>
                    </div>
                    <div>
                      <div
                        style="
                          flex: 1;
                          height: 90px;
                          background-color: #0c4f9c;
                          border-radius: 10px;
                          margin: 20px;
                          margin-bottom: 0px;
                          margin-top: 10px;
                        "
                      >
                        <div style="display: flex">
                          <div
                            style="
                              flex: 1;
                              width: 50px;
                              font-size: 15px;
                              color: white;
                              padding: 15px;
                              padding-right: 0px;
                              font-weight: bold;
                              padding-top: 10px;
                            "
                          >
                            <img src="/public/naohuan.png" style="width: 40px" />脑电帽
                          </div>
                          <el-button
                            type="success"
                            style="
                              margin-left: 10px;
                              width: 50px;
                              height: 40px;
                              borader: radius 20px;
                              background-color: white;
                              color: #464879;
                              font-size: 18px;
                              flex: 1.5;
                              margin-top: 30px;
                            "
                            @click="startAnalysis1"
                            >开始连接</el-button
                          >
                          <el-button
                            type="success"
                            style="
                              margin-left: 20px;
                              margin-right: 10px;
                              width: 50px;
                              height: 40px;
                              borader: radius 20px;
                              background-color: #154b8a;
                              font-size: 18px;
                              flex: 1.5;
                              margin-top: 30px;
                            "
                            @click="startAnalysis2"
                            >断开连接</el-button
                          >
                        </div>
                      </div>
                    </div>
                  </div>

                  <!--睡眠周期-->
                  <div style="flex: 1; border-radius: 10px">
                    <el-container class="all">
                      <el-main style="background-color: white">
                        <el-scrollbar>
                          <el-table
                            :data="tableData1"
                            stripe
                            style="width: 500px; height: 360px"
                            class="custom-table test"
                            :header-cell-style="{
                              background: '#0c4f9c',
                              color: 'white',
                            }"
                          >
                            <el-table-column type="index" width="55" />
                            <el-table-column prop="date" label="时间" width="300" />
                            <el-table-column
                              prop="status"
                              label="睡眠状态"
                              width="auto"
                            />
                          </el-table>
                        </el-scrollbar>
                      </el-main>
                    </el-container>
                  </div>
                </div>
              </el-footer>
            </el-container>
          </el-scrollbar>
        </el-main>
      </el-container>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted, reactive } from "vue";
import { ElTable } from "element-plus";
import echart2 from "../echarts/echart2.vue";
import axios from "axios";
import { ElMessage, ElMessageBox } from "element-plus";
import type { Action } from "element-plus";
import * as echarts from "echarts";

const tableData = ref([]);
const showSleepTimeColumn = false;
onMounted(() => {
  getList();
  var chartDom = document.getElementById("naodianshuju");
  var myChart = echarts.init(chartDom);
  myChart.setOption(option1);
  console.log(xAxis);
});
//脑电数据

const getList = async () => {
  var res = await axios.get("/adminapi/user");
  // console.log(res.data)
  tableData.value = res.data;
};

var multipleTableRef = ref();
var multipleSelection = ref([]);
const handleSelectionChange = (val) => {
  multipleSelection.value = val;
};

const toggleSelection = (rows) => {
  multipleTableRef.value.clearSelection();
};

const selectClick = (selection, row) => {
  if (selection.length > 1) {
    let del_row = selection.shift();
    multipleTableRef.value.toggleRowSelection(del_row, false); // 用于多选表格，切换某一行的选中状态，如果使用了第二个参数，则是设置这一行选中与否（selected 为 true 则选中）
  }
};

const isMessageBoxOpen = ref(false); // 用于跟踪提示对话框的显示状态

interface SelectionItem {
  name: string;
  upload: string;
  // 其他属性...
}

const startAnalysis = () => {
  // 如果提示对话框已经显示，不执行任何操作
  if (isMessageBoxOpen.value) {
    return;
  }
  const selectedNames = ref<SelectionItem[]>([]);
  selectedNames.value = multipleSelection.value;
  isMessageBoxOpen.value = true; // 设置提示对话框为已显示状态
  var names = selectedNames.value.map((item) => item.name);
  var namesString = names.join(", "); // 将获取到的名字数组转换成字符串
  var uploads = selectedNames.value.map((item) => item.upload);
  var uploadString = uploads.join(", ");
  var uploadType;
  beginPredict(uploadString);
  uploadType = uploadString.substr(-4);
  uploadString = uploadString.slice(0, -4);

  console.log("leixin" + uploadType);
  console.log(uploadString);
  if (multipleSelection.value.length > 0) {
    // 如果有选择项，才执行增加 percentage2 的逻辑
    ElMessageBox.alert(`当前被试对象为：${namesString}`, "提示").then(() => {
      isMessageBoxOpen.value = false; // 当提示对话框关闭时，重置显示状态
    });
  } else {
    // 如果没有选择项，可以显示提示或采取其他操作
    ElMessageBox.alert("请先选择被试对象", "提示").then(() => {
      isMessageBoxOpen.value = false; // 当提示对话框关闭时，重置显示状态
    });
  }
  let i = 1;
  console.log("文件名" + uploads);

  const intervalId = setInterval(() => {
    if (i < 1000) {
      getEEG(i, uploadString, uploadType);
      getStage(uploadString);
      console.log("执行" + i);
      i = i + 10;
    } else {
      clearInterval(intervalId); // 达到 500 次后清除间隔
    }
  }, 1000); // 每 1 秒执行一次
};

/*

*/

const startAnalysis1 = async () => {
  var re = await axios.post("/adminapi/bao1", { aa: 1 });
  let r = 1;
  ElMessageBox.alert(`脑电帽已连接`, "提示").then(() => {
          isMessageBoxOpen.value = false; // 当提示对话框关闭时，重置显示状态
        });
  const intervalId = setInterval(async () => {
    if (r > 0) {
      var re1 = await axios.post("/adminapi/bao3", { switch: 3 });
      console.log(re1.data);
      console.log(re1.data.data);
      if (re1.data.data === 0) {
        console.log("脑环未连接");
      } else if (re1.data.data === 1) {
        console.log("脑环已连接");

        setTimeout(() => {
          naoEEG(r);
          setTimeout(() => {
            naoStage(r);
          }, 5000);
        }, 1000); // 延迟 8 秒调用 naoEEG 和 naoStage
      } else {
        console.log("脑环断开连接");
      }
      r = r + 10;
    } else {
      clearInterval(intervalId); // 达到 500 次后清除间隔
    }
  }, 9000);
};

const startAnalysis2 = async () => {
  // 如果提示对话框已经显示，不执行任何操作
  if (isMessageBoxOpen.value) {
    return;
  }
  var re = await axios.post("/adminapi/bao2", { aa: 0 });
  ElMessageBox.alert(`脑电帽已断开`, "提示").then(() => {
    isMessageBoxOpen.value = false; // 当提示对话框关闭时，重置显示状态
    window.location.reload();
  });
};
var xAxis = ref([]);
var yAxis = ref([]);
//脑电数据获取函数
const getEEG = async (val, val1, val2) => {
  //脑电数值图数据
  var re = await axios.post("/adminapi/getEEG", {
    file: val1 + val2,
    select: 0,
    begin: val,
  });
  xAxis.value = re.data.data;
  option1.series[0].data = re.data.data;
  var re1 = await axios.post("/adminapi/getEEG", {
    file: val1 + val2,
    select: 1,
    begin: val,
  });

  // console.log("数据1=>",xAxis.value.arr);
  option1.series[1].data = re1.data.data;
  var chartDom = document.getElementById("naodianshuju");
  var myChart = echarts.init(chartDom);
  myChart.setOption(option1);
};

const option1 = {
  /*  tooltip: {
    trigger: 'axis', // 或 'item'
    formatter: function (params) {
        // 在这里添加工具提示格式化逻辑
        return params.name + ': ' + params.value;
    }
  },
*/
  legend: {
    data: ["EEG", "EOG"],
  },
  grid: {
    left: "3%",
    right: "4%",
    bottom: "3%",
    containLabel: true,
  },
  toolbox: {
    feature: {
      saveAsImage: {},
    },
  },
  //  tooltip: {
  // show: true, // 确保设置了此属性
  //},
  xAxis: {
    type: "category",
    boundaryGap: false,
    show: false,
    axisLabel: {
      interval: "auto", // 自动根据图表大小调整间隔
      rotate: -45, // 标签旋转角度
    },
  },
  yAxis: {
    type: "value",
    scale: true, // 根据数据自动调整最大最小值
  },
  series: [
    {
      name: "EEG",
      data: xAxis.value,
      type: "line",
      smooth: true,
      showSymbol: false,
      tooltip: {
        show: true, // 确保设置了此属性
      },
    },
    {
      name: "EOG",
      data: yAxis.value,
      type: "line",
      smooth: true,
      showSymbol: false,
      tooltip: {
        show: true, // 确保设置了此属性
      },
    },
  ],
};

//脑电状态
var tableData1 = ref([]);
const getStage = async (val) => {
  var stage = await axios.post("/adminapi/readStage", { file: val + ".npz", begin: 0 });
  console.log("stage的数值" + stage.data);
  // console.log(res.data)
  tableData1.value = stage.data.data;
};
//开始预测
const beginPredict = async (val) => {
  var predict = await axios.post("/adminapi/predict", { file: val });
};
const beginPredict2 = async (val) => {
  var predict = await axios.post("/adminapi/predict2", { file: val });
};
const getEEG2 = async (val, val1) => {
  //脑电数值图数据
  var re = await axios.post("/adminapi/getEEG", {
    file: val1 + ".npz",
    select: 0,
    begin: val,
  });
  xAxis.value = re.data.data;
  option1.series[0].data = re.data.data;
  var re1 = await axios.post("/adminapi/getEEG", {
    file: val1 + ".npz",
    select: 1,
    begin: val,
  });

  // console.log("数据1=>",xAxis.value.arr);
  option1.series[1].data = re1.data.data;
  var chartDom = document.getElementById("naodianshuju");
  var myChart = echarts.init(chartDom);
  myChart.setOption(option1);
};

const naoEEG = async (val) => {
  //脑电数值图数据
  var re = await axios.post("/adminapi/naoGetEEG", { select: 0, begin: val });
  xAxis.value = re.data.data;
  option1.series[0].data = re.data.data;
  var re1 = await axios.post("/adminapi/naoGetEEG", { select: 1, begin: val + 100 });

  // console.log("数据1=>",xAxis.value.arr);
  option1.series[1].data = re1.data.data;
  var chartDom = document.getElementById("naodianshuju");
  var myChart = echarts.init(chartDom);
  myChart.setOption(option1);
};
const naoStage = async (val) => {
  var stage = await axios.post("/adminapi/naoReadStage", { begin: val });
  console.log("stage的数值" + stage.data);
  // console.log(res.data)
  tableData1.value = stage.data.data;
};
</script>

<style scoped>
.all {
  border: 1px solid #ffffff;
}
.title {
  background: white;
  width: auto;
  float: left;
  text-align: left;
  display: block;
  padding: 10px;
  font-weight: bold;
  font-size: 20px;
}
.el-scrollbar__wrap::-webkit-scrollbar-track {
  background-color: #f0f0f0; /* 轨道的背景颜色 */
}

.demo-button-style {
  margin-top: 24px;
}
::v-deep .el-table th.el-table__cell:nth-child(1) .cell {
  visibility: hidden;
}

.el-row {
  margin-bottom: 20px;
}
.el-row:last-child {
  margin-bottom: 0;
}
.el-col {
  border-radius: 4px;
}

.grid-content {
  border-radius: 4px;
  background-color: #bbd3f0;
  width: 380px;
  height: 170px;
  flex: 1;
  margin: 10px;
  border-radius: 10px;
}

:deep(.el-table__header-wrapper) {
  border-radius: 8px;
  z-index: 100 !important;
  overflow: hidden;
  -webkit-border-radius: 8px;
  -moz-border-radius: 8px;
  -ms-border-radius: 8px;
  -o-border-radius: 8px;
}

:deep(.el-table td:first-child) {
  border-left: 1px solid #fff;
  border-radius: 8px 0 0 8px;
  padding: 2px;
  z-index: 999;
}

:deep(.el-table td:last-child) {
  border-right: 1px solid #fff;
  border-radius: 0 8px 8px 0;
  z-index: 999;
  padding: 2px;
}
.custom-table .el-table__fixed-left-wrapper,
.custom-table .el-table__fixed-right-wrapper {
  overflow: visible;
}
.custom-table::before,
.custom-table::after {
  content: "";
  position: absolute;
  top: 0;
  bottom: 0;
  width: 8px;
  z-index: 1;
}

.custom-table::before {
  left: 0;
  border-top-left-radius: 10px;
  border-bottom-left-radius: 10px;
}
.test::before {
  left: 0;

  border-top-left-radius: 10px;
  border-bottom-left-radius: 10px;
}

.custom-table::after {
  right: 0;
  background-color: #fff;
  border-radius: 10px;
}

.test::after {
  right: 0;
  background-color: #fff;
  border-radius: 10px;
}
.control-table {
  position: relative;
}

.test.el-table--striped .el-table__body tr.el-table__row--striped td {
  background-color: #292a84;
  color: #181eb9;
}

.test ::v-deep .el-table__row {
  background-color: #a2c5f1;
}

::v-deep .el-table__header {
  min-width: 296px !important;
}
.test.el-table {
  color: #000000;
}
::v-deep .custom-table .el-table {
  background-color: #7678af !important;
}
.el-table {
  --el-table-tr-bg-color: transparent;
}
.el-button--success {
  --el-button-border-color: #b6cad7;
  --el-button-hover-border-color: #b6cad7;
}
</style>
