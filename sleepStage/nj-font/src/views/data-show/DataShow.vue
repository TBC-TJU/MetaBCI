<template>
  <div class="bg" style="padding: 20px">
    <div
      class="ti"
      style="position: fixed; top: 10px; margin-left: 0px; color: white; font-size: 30px"
    >
      睡眠报告
    </div>
    <div class="common-layout">
      <el-container style="border: 0px">
        <!--选择被试-->
        <div
          style="
            display: flex;
            height: 420px;
            width: 300px;
            margin-left: 10px;
            border-radius: 10px;
            border: 0px;
          "
        >
          <el-container class="all" style="width: auto">
            <el-header class="title" style="height: 40px; padding: 10px"
              >选择被试</el-header
            >
            <el-main style="background-color: white; padding: 10px">
              <div>
                <el-table
                  ref="multipleTableRef"
                  :data="tableData1"
                  style="height: 340px; width: 270px"
                  @select="selectClick"
                  stripe
                  @select-all="clickselect"
                  @selection-change="handleSelectionChange"
                  class="custom-table test"
                  :header-cell-style="{ background: '#0c4f9c', color: 'white' }"
                >
                  <el-table-column type="selection" width="55" />
                  <el-table-column
                    prop="id"
                    label="ID"
                    width="80"
                    v-if="showSleepTimeColumn"
                  />
                  <el-table-column label="上传日期" width="120">
                    <template #default="scope">{{ scope.row.date }}</template>
                  </el-table-column>
                  <el-table-column property="name" label="名称" width="80" />
                </el-table>
              </div>
            </el-main>
          </el-container>
        </div>
        <!--睡眠分期时相图-->
        <el-main style="padding-left: 20px; padding-top: 0px">
          <div style="display: flex; height: 40px">
            <div style="font-weight: bold; font-size: 20px; width: 60px">姓名：</div>
            <div
              v-if="tableData.length > 0"
              style="
                margin: 0px;
                padding: 0px;
                float: right;
                font-weight: bold;
                color: #0c4f9c;
                font-size: 20px;
              "
            >
              <p>{{ tableData1[tableData[0].id - 1].name }}</p>
            </div>
          </div>
          <div
            style="
              display: flex;
              margin: 0px;
              margin-left: 20px;
              padding: 10px;
              height: 360px;
              width: 850px;
              background-color: white;
              border-radius: 20px;
            "
          >
            <e-charts id="shixiang" class="chart" :option="option1" />
          </div>
        </el-main>
      </el-container>
    </div>
    <!--分期结果-->
    <div
      style="
        background-color: white;
        width: 1200px;
        margin-left: 10px;
        border-radius: 10px;
      "
    >
      <div
        style="
          background: white;
          width: auto;
          text-align: left;
          display: block;
          padding: 10px;
          font-weight: bold;
          font-size: 20px;
          border-radius: 10px;
        "
      >
        分期结果
      </div>
      <div style="display: flex">
        <div style="flex: 1" class="zhouqi">
          <div style="font-weight: bold; padding-top: 10px">W期</div>
          <div style="font-weight: bold; padding-top: 10px">清醒期</div>
          <div style="display: flex">
            <div style="flex: 2">
              <img src="/public/bofang.png" style="width: 60px; margin: 20px" />
            </div>
            <div
              style="
                flex: 3;
                padding-top: 20px;
                margin: 0px;
                width: 100px;
                text-align: left;
                font-weight: bold;
              "
            >
              <div style="display: block">时间:</div>
              <div
                v-if="tableData.length > 0"
                style="
                  float: left;
                  margin: 0px;
                  padding: 0px;
                  font-size: 30px;
                  font-weight: bold;
                  color: #0c4f9c;
                "
              >
                <p>{{ tableData[0].wtime }}</p>
              </div>
            </div>
          </div>
        </div>
        <div style="flex: 1" class="zhouqi">
          <div style="font-weight: bold; padding-top: 10px">N1期</div>
          <div style="font-weight: bold; padding-top: 10px">非快速眼动Ⅰ期</div>
          <div style="display: flex">
            <div style="flex: 2">
              <img src="/public/bofang1.png" style="width: 60px; margin: 20px" />
            </div>
            <div
              style="
                flex: 3;
                padding-top: 20px;
                margin: 0px;
                width: 100px;
                text-align: left;
                font-weight: bold;
              "
            >
              <div style="display: block">时间:</div>
              <div
                v-if="tableData.length > 0"
                style="
                  float: left;
                  margin: 0px;
                  padding: 0px;
                  font-size: 30px;
                  font-weight: bold;
                  color: #0c4f9c;
                "
              >
                <p>{{ tableData[0].n1time }}</p>
              </div>
            </div>
          </div>
        </div>
        <div style="flex: 1" class="zhouqi">
          <div style="font-weight: bold; padding-top: 10px">N2期</div>
          <div style="font-weight: bold; padding-top: 10px">非快速眼动Ⅱ期</div>
          <div style="display: flex">
            <div style="flex: 2">
              <img src="/public/bofang.png" style="width: 60px; margin: 20px" />
            </div>
            <div
              style="
                flex: 3;
                padding-top: 20px;
                margin: 0px;
                width: 100px;
                text-align: left;
                font-weight: bold;
              "
            >
              <div style="display: block">时间:</div>
              <div
                v-if="tableData.length > 0"
                style="
                  float: left;
                  margin: 0px;
                  padding: 0px;
                  font-size: 30px;
                  font-weight: bold;
                  color: #0c4f9c;
                "
              >
                <p>{{ tableData[0].n2time }}</p>
              </div>
            </div>
          </div>
        </div>
        <div style="flex: 1" class="zhouqi">
          <div style="font-weight: bold; padding-top: 10px">N3期</div>
          <div style="font-weight: bold; padding-top: 10px">非快速眼动Ⅲ期</div>
          <div style="display: flex">
            <div style="flex: 2">
              <img src="/public/bofang1.png" style="width: 60px; margin: 20px" />
            </div>
            <div
              style="
                flex: 3;
                padding-top: 20px;
                margin: 0px;
                width: 100px;
                text-align: left;
                font-weight: bold;
              "
            >
              <div style="display: block">时间:</div>
              <div
                v-if="tableData.length > 0"
                style="
                  float: left;
                  margin: 0px;
                  padding: 0px;
                  font-size: 30px;
                  font-weight: bold;
                  color: #0c4f9c;
                "
              >
                <p>{{ tableData[0].n3time }}</p>
              </div>
            </div>
          </div>
        </div>
        <div style="flex: 1; background-color: #00286d; color: white" class="zhouqi">
          <div style="font-weight: bold; padding-top: 10px">REM期</div>
          <div style="font-weight: bold; padding-top: 10px">快速眼动期</div>
          <div style="display: flex">
            <div style="flex: 2">
              <img src="/public/bofang2.png" style="width: 60px; margin: 20px" />
            </div>
            <div
              style="
                flex: 3;
                padding-top: 20px;
                margin: 0px;
                width: 100px;
                text-align: left;
                font-weight: bold;
              "
            >
              <div style="display: block">时间:</div>
              <div
                v-if="tableData.length > 0"
                style="
                  float: left;
                  margin: 0px;
                  padding: 0px;
                  font-size: 30px;
                  font-weight: bold;
                  color: #9dc5f3;
                "
              >
                <p>{{ tableData[0].remtime }}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div style="display: flex; width: 150px">
      <div style="margin: 30px; margin-right: 0px; flex: 1; width: 150px">
        <div style="display: flex; width: 150px; margin-bottom: 50px">
          <div style="flex: 1; width: 50px; height: 50px; margin-right: 0px">
            <img
              src="/public/plane1.png"
              style="width: 50px; height: 50px; margin-right: 0px"
            />
          </div>
          <div style="flex: 3; margin-left: 10px">
            <div
              style="
                float: left;
                text-align: left;
                display: block;
                padding: 0px;
                font-weight: bold;
              "
            >
              睡眠总时长
            </div>
            <div>
              <div
                v-if="tableData.length > 0"
                style="
                  margin: 0px;
                  padding: 0px;
                  float: right;
                  font-weight: bold;
                  color: #0c4f9c;
                  font-size: 25px;
                  width: 150px;
                "
              >
                <p>{{ tableData[0].sleeptime }}</p>
              </div>
            </div>
          </div>
        </div>
        <div style="display: flex; width: 150px">
          <div style="flex: 1; width: 50px; height: 50px; margin-right: 0px">
            <img src="/public/woshou1.png" style="width: 50px" />
          </div>
          <div style="flex: 3; margin-left: 10px">
            <div
              style="
                float: left;
                text-align: left;
                display: block;
                margin: 0px;
                font-weight: bold;
              "
            >
              评估类型
            </div>
            <div>
              <div
                v-if="tableData.length > 0"
                style="
                  margin: 0px;
                  padding: 0px;
                  float: right;
                  font-weight: bold;
                  color: #0c4f9c;
                  font-size: 25px;
                  width: 150px;
                "
              >
                <p>{{ tableData[0].type }}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div style="felx: 1; margin: 10px; border-radius: 20px">
        <e-charts
          id="zhuzhuangtu"
          class="chart1"
          :option="option2"
          style="
            width: 500px;
            height: 300px;
            background-color: white;
            border-radius: 20px;
          "
        ></e-charts>
      </div>
      <div style="felx: 1; margin: 10px; border-radius: 20px">
        <e-charts
          id="shanxin"
          style="
            width: 500px;
            height: 300px;
            background-color: white;
            border-radius: 20px;
          "
          :option="option3"
        ></e-charts>
      </div>
    </div>
  </div>
</template>
<script lang="ts" setup>
import { ref, onMounted } from "vue";
import type { TabsPaneContext } from "element-plus";
import axios from "axios";
import echart1 from "../echarts/echart1.vue";
import echart3 from "../echarts/echart3.vue";
import echart4 from "../echarts/echart4.vue";
import * as echarts from "echarts";
import { table } from "console";

interface User {
  date: string;
  name: string;
}

const activeName = ref("first");
const multipleTableRef = ref();
const multipleSelection = ref([]);
const handleClick = (tab: TabsPaneContext, event: Event) => {
  console.log(tab, event);
};

const tableData = ref([] as any[]);
const tableData1 = ref([] as any[]);
const showSleepTimeColumn = false;

onMounted(() => {
  getList();
});

const getList = async () => {
  var res1 = await axios.get("/adminapi/user");
  // console.log(res.data)
  tableData1.value = res1.data;
};

interface SelectionItem {
  name: string;
  upload: string;
  // 其他属性...
}

const selectClick = (selection, row) => {
  if (selection.length > 1) {
    let del_row = selection.shift();
    multipleTableRef.value.toggleRowSelection(del_row, false); // 用于多选表格，切换某一行的选中状态，如果使用了第二个参数，则是设置这一行选中与否（selected 为 true 则选中）
  }

  // 获取第一个选中项的ID
  const selectedId = selection[0]?.id;
  // console.log(selectedId);
  // 检查是否有选中项
  if (selectedId) {
    axios
      .get(`/adminapi/history/${selectedId}`)
      .then((response) => {
        tableData.value = response.data;
        changeEchart();
      })
      .catch((error) => {
        console.error(error);
      });
    console.log(tableData.value);
  } else {
    // 如果没有选中项，将 tableData 清空
    tableData.value = [];
  }
};
const clickselect = (value) => {
  //multipleTableRef.value.clearSelection()
};
//选中
const handleSelectionChange = (val) => {
  multipleSelection.value = val;
  //11.20的bug
  const selectedNames = ref<SelectionItem[]>([]);
  selectedNames.value = multipleSelection.value;
  var uploads = selectedNames.value.map((item) => item.upload);
  var uploadString = uploads.join(", ");
  uploadString = uploadString.slice(0, -4);
  getNPZ(uploadString);
};

const cellClass = (row, column) => {
  if (column.index === 0) {
    return "disabledCheck";
  }
};

const getNPZ = async (val) => {
  var re = await axios.post("/adminapi/readNPZ", {
    file: val + ".npz",
    select: 1,
    begin: 10,
  });
  option1.series[0].data = re.data.data;
  var chartDom = document.getElementById("shixiang");
  var myChart = echarts.init(chartDom);
  myChart.setOption(option1);
};

const option1 = {
  title: {
    text: "睡眠分期时相图",
  },
  /*tooltip: {
    trigger: 'axis'
  },*/
  legend: {
    data: ["显示折线"],
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
  xAxis: {
    type: "category",
    show: false,
    axisLabel: {
      interval: 1, // 或者设置为 1，根据需要调整
      // 其他配置...
    },
  },
  yAxis: {
    type: "category", // 使用类别型坐标轴
    data: ["W", "N1", "N2", "N3", "R"],
    axisLabel: {
      formatter: function (value) {
        switch (value) {
          case "0":
            return "W";
          case "1":
            return "N1";
          case "2":
            return "N2";
          case "3":
            return "N3";
          case "4":
            return "R";
          default:
            return value;
        }
      },
    },
    // 刻度标签数据
    splitLine: {
      show: true, // 显示分隔线
      axisTick: {
        alignWithLabel: true, // 使刻度线对齐标签
        show: false, // 不显示刻度线
      },
      lineStyle: {
        type: "dashed", // 设置分隔线为虚线
        opacity: 100, // 设置分隔线的透明度
        // 可以设置其他样式属性
      },
    },
  },

  series: [
    {
      name: "显示折线",
      type: "line",
      step: "start",
      data: [],
      itemStyle: {
        color: "#0c4f9c",
        opacity: 0, // 将点的透明度设置为0，即隐藏点
      },
    },
  ],
  dataZoom: [
    // 这个dataZoom组件，若未设置xAxisIndex或yAxisIndex，则默认控制x轴。
    {
      type: "slider", //这个 dataZoom 组件是 slider 型 dataZoom 组件（只能拖动 dataZoom 组件导致窗口变化）
      xAxisIndex: 0, //控制x轴
      start: 0, // 左边在 10% 的位置
      end: 25, // 右边在 60% 的位置
    },
    {
      type: "inside", //这个 dataZoom 组件是 inside 型 dataZoom 组件（能在坐标系内进行拖动，以及用滚轮（或移动触屏上的两指滑动）进行缩放）
      xAxisIndex: 0, //控制x轴
      start: 10,
      end: 60,
    },
    {
      type: "inside", // inside 型 dataZoom 组件
      yAxisIndex: 0, //控制y轴
      start: 0,
      end: 100,
    },
  ],
};

const changeEchart = async () => {
  var chartDom1 = document.getElementById("zhuzhuangtu");
  option2.series[0].data = [];
  //  console.log("数据为" + tableData.value[0]?.wtime); // 这个是为了确认数据是否存在，可选链运算符可以避免空值导致的错误
  option2.series[0].data.push(parseFloat((tableData.value[0]?.wtime).slice(0, -1)));
  option2.series[0].data.push(parseFloat((tableData.value[0]?.n1time).slice(0, -1)));
  option2.series[0].data.push(parseFloat((tableData.value[0]?.n2time).slice(0, -1)));
  option2.series[0].data.push(parseFloat((tableData.value[0]?.n3time).slice(0, -1)));
  option2.series[0].data.push(parseFloat((tableData.value[0]?.remtime).slice(0, -1)));
  var myChart1 = echarts.init(chartDom1);
  option2 && myChart1.setOption(option2);
  //shanxin
  var chartDom2 = document.getElementById("shanxin");
  option3.series[0].data = [];
  option3.series[0].data.push({
    value: parseFloat((tableData.value[0]?.wtime).slice(0, -1)),
    name: "W期",
  });
  option3.series[0].data.push({
    value: parseFloat((tableData.value[0]?.n1time).slice(0, -1)),
    name: "N1期",
  });
  option3.series[0].data.push({
    value: parseFloat((tableData.value[0]?.n2time).slice(0, -1)),
    name: "N2期",
  });
  option3.series[0].data.push({
    value: parseFloat((tableData.value[0]?.n3time).slice(0, -1)),
    name: "N3期",
  });
  option3.series[0].data.push({
    value: parseFloat((tableData.value[0]?.remtime).slice(0, -1)),
    name: "REM期",
  });
  console.log(option3.series[0].data);
  var myChart2 = echarts.init(chartDom2);
  option3 && myChart2.setOption(option3);
};

let colors = ["#e4e6ff", "#9fa3f7", "#C4C6F5", "#E3C4F7", "#8CA6E2"];
const option2 = {
  xAxis: {
    type: "category",
    data: ["W", "N1", "N2", "N3", "REM"],
  },
  yAxis: {
    type: "value",
  },
  series: [
    {
      color: colors,
      data: [] as number[],
      type: "bar",
      itemStyle: {
        normal: {
          color: function (params) {
            let idx = params.dataIndex;
            return colors[idx];
          },
          barBorderRadius: [50, 50, 0, 0], // 左上，右上，右下，左下
        },
      },
    },
  ],
};

interface DataItem {
  value: number;
  name: string;
}

const option3 = {
  legend: {
    top: "bottom",
  },
  toolbox: {
    show: true,
    feature: {
      mark: { show: true },
      dataView: { show: true, readOnly: false },
      restore: { show: true },
      saveAsImage: { show: true },
    },
  },
  series: [
    {
      name: "Nightingale Chart",
      type: "pie",
      radius: [20, 100],
      center: ["50%", "50%"],
      roseType: "area",
      itemStyle: {
        borderRadius: 8,
      },
      data: [] as DataItem[],
    },
  ],
};
</script>
<style scoped>
.demo-tabs > .el-tabs__content {
  padding: 22px;
  color: #6b778c;
  font-size: 32px;
  font-weight: 400;
}
.echarts {
  height: 500px;
  width: 800px;
}
.percentage-value {
  display: block;
  margin-top: 10px;
  font-size: 28px;
}
.percentage-label {
  display: block;
  margin-top: 10px;
  font-size: 12px;
}
.container {
  display: flex;
  justify-content: space-between; /* 左右对齐 */
  align-items: center; /* 垂直居中 */
}

.right-container {
  display: flex;
  flex-direction: column; /* 垂直方向排列 */
}
.all {
  border: 1px solid #cccccc;
}
::v-deep .el-table th.el-table__cell:nth-child(1) .cell {
  visibility: hidden;
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
  border-left: 1px solid #e2ecfa;
  border-radius: 8px 0 0 8px;
  padding: 2px;
  z-index: 999;
}

:deep(.el-table td:last-child) {
  border-right: 1px solid #e2ecfa;
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
  background-color: #a9aad1;
  color: #7678af;
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
.el-table {
  --el-table-tr-bg-color: transparent;
}
.zhouqi {
  background-color: transparent;
  border: 2px solid rgb(154, 154, 154);
  height: 150px;
  margin: 20px;
  text-align: center;
  justify-content: space-between; /* 左右对齐 */
  align-items: center; /* 垂直居中 */
  border-radius: 10px;
}
.chart {
  height: 350px;
  width: 800px;
  padding: 0px;
  background-color: white;
  margin-left: 20px;
  border-radius: 20px;
}
</style>
