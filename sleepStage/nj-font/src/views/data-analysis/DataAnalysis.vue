<template>
  <div class="common-layout">
    <el-container>
      <el-aside width="200px">
        <el-scrollbar>
          <div>
            <el-container class="all" style="margin-bottom:30px;">
              <el-header class="title" style=" height: 50px;">被试名称</el-header>
              <el-main> <el-checkbox v-model="checked1" label="张三" size="large" /></el-main>
            </el-container>
          </div>
          <div>
            <el-container class="all">
              <el-header class="title" style=" height: 50px;">数据类型</el-header>
              <el-main> <el-checkbox v-model="checked2" label="EEG" size="large" /></el-main>
            </el-container>
          </div>
        </el-scrollbar>
      </el-aside>
      <el-main>
        <el-scrollbar style="height: 400px;">
          <div>
            <el-container class="all" style="margin-bottom:30px;margin-top:0px;">
              <el-header class="title" style=" height: 50px;">已有数据</el-header>
              <el-main>
                <el-table
                ref="multipleTableRef"
                :data="tableData"
                style="width: 100%"
                @selection-change="handleSelectionChange"
              >
                <el-table-column type="selection" width="55" />
                <el-table-column property="name" label="数据类型" width="120" />
                <el-table-column property="type" label="数据类型" width="120" />
                <el-table-column property="number" label="数据编号" width="120" />
                <el-table-column property="result" label="已有结果" width="120" />
              </el-table>
              </el-main>
            </el-container>
          </div>
          <div>
            <el-container class="all">
              <el-header class="title" style=" height: 50px;">分析方法</el-header>
              <el-main> <div>
                <el-checkbox-group v-model="checkboxGroup1" size="large">
                  <el-checkbox-button v-for="way in ways" :key="way" :label="way">
                    {{ way }}
                  </el-checkbox-button>
                </el-checkbox-group>
              </div>
            </el-main>
            </el-container>
          </div>
        </el-scrollbar>
        <div>
          <el-button type="success">开始分析</el-button>
        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script lang="ts" setup>
import { ref } from 'vue'
import { ElTable } from 'element-plus'
const checked1 = ref()
const checked2 = ref()
const checkboxGroup1 = ref(['预处理'])
const ways = ['预处理', '构建相位同步网络', '网络模块划分', '模块化参数计算','功率谱分析','预测评估']
interface User {
  name: string
  type:string
  number:string
  result:string
}

const multipleTableRef = ref<InstanceType<typeof ElTable>>()
const multipleSelection = ref<User[]>([])
const handleSelectionChange = (val: User[]) => {
  multipleSelection.value = val
}

const tableData: User[] = [
  {
    name: "张三",
    type:"EEG",
    number:"20181017",
    result:"原始数据"
  }, 
  {
    name: "李四",
    type:"EOG",
    number:"20181010",
    result:"原始数据"
  }
]

</script>

<style scoped>
.all{
  border: 1px solid #CCCCCC;
}
.title{
  background: rgb(217, 224, 224);
  width: auto;
  display: flex;
  align-items: center;
  justify-content: center;
}
.el-scrollbar__wrap::-webkit-scrollbar-track {
  background-color: #f0f0f0; /* 轨道的背景颜色 */
}

.demo-button-style {
  margin-top: 24px;
}
</style>
