<template class="bg" >
  <div class="bg" style="height: 700px;">
    <div class="ti" style="position: fixed;top:10px;margin-left:10px;color: white; font-size: 30px;">历史数据</div>
    <div  style="width: 1000px;padding: 30px;height:450px;padding-left:150px;border-radius:10px;" >
      <el-table ref="tableRef" row-key="id" :data="tableData" 
      style="height: 450px;border-radius:10px;border: 5px solid #0c4f9c;color:black;" stripe :header-cell-style="{background: '#0c4f9c',color:'white','text-align':'center'}" :cell-style="{'text-align':'center'} " class="test">
      <el-table-column type="index" width="100"/>
      <el-table-column
        prop="date"
        label="上传日期"
        sortable
        width="300"
        column-key="date"
        :filters="dateFilters"
        :filter-method="filterHandler"
      />
      <el-table-column prop="name" label="名称" width="350" />
      <el-table-column
        prop="tag"
        label="是否测试"
        width="auto"
        :filters="[
          { text: '是', value: 'YES' },
          { text: '否', value: 'NO' },
        ]"
        :filter-method="filterTag"
        filter-placement="bottom-end"
      >
        <template #default="scope">
          <el-tag
            :type="scope.row.tag === 'YES' ? '' : 'success'"
            disable-transitions
            >{{ scope.row.tag }}</el-tag
          >
        </template>
      </el-table-column>
    </el-table>
    </div>
    <div style="margin-left: 500px;padding-bottom:20px;">
      <el-button @click="resetDateFilter" style="background-color: #0c4f9c;color:white;font-weight:bold;font-size:20px;height: 40px;">重置日期筛选</el-button>
      <el-button @click="clearFilter" style="background-color: white;color:#0c4f9c;font-weight:bold;font-size:20px;height: 40px;">重置全部筛选</el-button>
    </div>
    
  </div>
  
</template>

<script lang="ts" setup>
import { ref, onMounted } from 'vue'
import type { TableColumnCtx, TableInstance } from 'element-plus'
import axios from 'axios';

interface User {
  date: string
  name: string
  tag: string
  id:number
}

interface TableDataItem {
  date: string;
  name: string
  tag: string
  id:number
}


const tableData = ref<TableDataItem[]>([]);

onMounted(() => {
  getList()
})

const getList = async () => {
  try {
    const res = await axios.get("/adminapi/user");
    tableData.value = res.data;
    updateDateFilters(); // 在数据加载后立即更新筛选器
  } catch (error) {
    console.error("Error fetching data:", error);
  }
};


const dateFilters = ref<Array<{ text: string; value: string; }>>([]);

const updateDateFilters = () => {
  const uniqueDates = new Set<string>();

  // 获取所有不重复的日期值
  for (const item of tableData.value) {
    uniqueDates.add(item.date);
  }

  // 将日期值转换为筛选器格式
  const newDateFilters = Array.from(uniqueDates).map(date => ({
    text: date,
    value: date,
  }));

  // 更新外部的 dateFilters 变量
  dateFilters.value = newDateFilters;

  // 找到日期列的元素并更新筛选器选项
  const dateColumn = document.querySelector('.el-table-column--date');
  if (dateColumn) {
    dateColumn.setAttribute(':filters', JSON.stringify(newDateFilters));
  }
};



const tableRef = ref<TableInstance>()

const resetDateFilter = () => {
  tableRef.value!.clearFilter(['date'])
}
// TODO: improvement typing when refactor table
const clearFilter = () => {
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  tableRef.value!.clearFilter()
}

const filterTag = (value: string, row: User) => {
  return row.tag === value
}
const filterHandler = (
  value: string,
  row: User,
  column: TableColumnCtx<User>
) => {
  const property = column['property']
  return row[property] === value
}

const tableRowClassName=({ rowIndex }) =>{
      if (rowIndex % 2 === 0) {
        return "warning-row";
      } else if (rowIndex % 2 === 1) {
        return "success-row";
      }
      return "";
    }


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
::v-deep.el-table tbody tr:hover > td {
  background-color: transparent !important;
}
</style>
