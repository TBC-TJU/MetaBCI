<template>
  <div class="bg" style="height: 700px">
    <div
      class="ti"
      style="position: fixed; top: 10px; margin-left: 10px; color: white; font-size: 30px"
    >
      历史报告
    </div>
    <div
      style="
        width: 1200px;
        padding: 30px;
        height: 450px;
        padding-left: 80px;
        border-radius: 10px;
      "
    >
      <el-table
        ref="tableRef"
        row-key="id"
        :data="tableData"
        class="test"
        style="
          height: 450px;
          border-radius: 10px;
          border: 5px solid #0c4f9c;
          color: black;
        "
        stripe
        :header-cell-style="{
          background: '#0c4f9c',
          color: 'white',
          'text-align': 'center',
        }"
        :cell-style="{ 'text-align': 'center' }"
      >
        <el-table-column type="index" width="50" />
        <el-table-column type="expand">
          <template #default="props" style="margin: 0px; padding: 0px">
            <div
              m="4"
              style="background-color: white; color: #0c4f9c; margin: 0px; padding: 0px"
            >
              <h2>详细数据</h2>
              <el-table :data="props.row.detailData">
                <el-table-column prop="date" label="采集时间" />
                <el-table-column prop="time" label="记录时长" />
                <el-table-column prop="sleeptime" label="睡眠时长" />
                <el-table-column prop="wtime" label="W期时长" />
                <el-table-column prop="n1time" label="N1期时长" />
                <el-table-column prop="n2time" label="N2期时长" />
                <el-table-column prop="n3time" label="N3期时长" />
                <el-table-column prop="remtime" label="REM期时长" />
                <el-table-column prop="wrate" label="W期比例" />
                <el-table-column prop="n1rate" label="N1期比例" />
                <el-table-column prop="n2rate" label="N2期比例" />
                <el-table-column prop="n3rate" label="N3期比例" />
                <el-table-column prop="remrate" label="REM期比例" />
                <el-table-column prop="type" label="患者类型" />
              </el-table>
            </div>
          </template>
        </el-table-column>
        <el-table-column
          prop="date"
          label="上传日期"
          sortable
          width="180"
          column-key="date"
          :filters="dateFilters"
          :filter-method="filterHandler"
        />
        <el-table-column prop="name" label="名称" width="180" />
        <el-table-column prop="type" label="数据类型" width="180" />
        <el-table-column prop="content" label="内容" :formatter="formatter" />
      </el-table>
    </div>

    <div style="margin-left: 500px; padding-bottom: 20px">
      <el-button
        @click="resetDateFilter"
        style="
          background-color: #0c4f9c;
          color: white;
          font-weight: bold;
          font-size: 20px;
          height: 40px;
        "
        >重置日期筛选</el-button
      >
      <el-button
        @click="clearFilter"
        style="
          background-color: white;
          color: #0c4f9c;
          font-weight: bold;
          font-size: 20px;
          height: 40px;
        "
        >重置全部筛选</el-button
      >
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import type { TableColumnCtx, TableInstance } from "element-plus";
import axios from "axios";

interface User {
  showDetail: any;
  detailData: any;
  date: string; // 确保此处有 date 属性
  name: string;
  content: string;
  id: number;
  type: string;
}

const tableData = ref<User[]>([]);

onMounted(() => {
  getList();
  tableData.value.forEach((item) => {
    toggleDetail(item);
  });
});

const getList = async () => {
  var res1 = await axios.get("/adminapi/user");
  for (const item of res1.data) {
    item.showDetail = true; // 用于控制是否显示详细数据，默认为true，因为你希望默认展开
    item.detailData = await getDetailData(item.id);
  }
  tableData.value = res1.data;
  updateDateFilters();
};

const dateFilters = ref<Array<{ text: string; value: string }>>([]);

const updateDateFilters = () => {
  const uniqueDates = new Set<string>();

  // 获取所有不重复的日期值
  for (const item of tableData.value) {
    uniqueDates.add(item.date);
  }

  // 将日期值转换为筛选器格式
  const newDateFilters = Array.from(uniqueDates).map((date) => ({
    text: date,
    value: date,
  }));

  // 更新外部的 dateFilters 变量
  dateFilters.value = newDateFilters;

  // 找到日期列的元素并更新筛选器选项
  const dateColumn = document.querySelector(".el-table-column--date");
  if (dateColumn) {
    dateColumn.setAttribute(":filters", JSON.stringify(newDateFilters));
  }
};

const getDetailData = async (id) => {
  try {
    const response = await axios.get(`/adminapi/history/${id}`);
    return response.data;
  } catch (error) {
    console.error(error);
    return [];
  }
};

const tableRef = ref<TableInstance>();

const resetDateFilter = () => {
  tableRef.value!.clearFilter(["date"]);
};
// TODO: improvement typing when refactor table
const clearFilter = () => {
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  tableRef.value!.clearFilter();
};
const formatter = (row: User, column: TableColumnCtx<User>) => {
  return row.content;
};

const filterHandler = (value: string, row: User, column: TableColumnCtx<User>) => {
  const property = column["property"];
  return row[property] === value;
};

const parentBorder = ref(false);
const childBorder = ref(false);

const toggleDetail = (row: User) => {
  // 如果当前行的 showDetail 为 true，则打开详细数据
  if (row.showDetail) {
    row.showDetail = true;
    // 在这里使用 axios 或其他方式加载详细数据，然后更新 detailData 属性
    // 例如，假设要加载对应 ID 的详细数据
    axios
      .get(`/adminapi/history/${row.id}`)
      .then((response) => {
        row.detailData = response.data;
      })
      .catch((error) => {
        console.error(error);
      });
  }
};
</script>

<style scoped>
.test :deep.el-table--striped .el-table__body tr.el-table__row--striped td {
  background: #a2c5f1;
  color: #000000;
}

/* 斑马纹颜色定义完之后会显示自带的边框颜色，这里要重置 */
:deep.el-table td,
.building-top .el-table th.is-leaf {
  border: none;
}
/* 禁用鼠标悬停时的颜色变化 */
</style>
