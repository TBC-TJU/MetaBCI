<template>
  <el-aside width="200px" style="background-color: #0c4f9c; color: aliceblue">
    <el-scrollbar>
      <div style="font-size: 20px; font-weight: bold; margin-bottom: 0px;margin-left:30px;" class="ti">
        NeuroDoc
      </div>
      <!--居中-->
      <div style="font-size: 16px; margin: 10px; margin-top: 0px">
        意识障碍睡眠分期和评估
      </div>
      <div>
        <el-menu
          :default-active="route.fullPath"
          :router="true"
          style="margin: 0px"
          background-color="#155398"
          text-color="white"
        >
          <template v-for="data in datalist" :key="data.path">
            <div>
              <!--扩展列表-->
              <el-sub-menu
                :index="data.path"
                v-if="data.children.length && checkAuth(data.path)"
              >
                <template #title>
                  <el-icon>
                    <component :is="mapIcons[data.icon]"></component>
                  </el-icon>
                  <span>{{ data.title }}</span>
                </template>
                <template v-for="item in data.children" :key="item.path">
                  <el-menu-item :index="item.path" v-if="checkAuth(item.path)">
                    <el-icon>
                      <component :is="mapIcons[item.icon]"></component>
                    </el-icon>
                    <span>{{ item.title }}</span>
                  </el-menu-item>
                </template>
              </el-sub-menu>
              <!--单一列表-->
              <el-menu-item
                :index="data.path"
                v-else-if="checkAuth(data.path)"
                style="color: white"
              >
                <el-icon>
                  <component :is="mapIcons[data.icon]"></component>
                </el-icon>
                <span>{{ data.title }}</span>
              </el-menu-item>
            </div>
          </template>
        </el-menu>
      </div>
    </el-scrollbar>
  </el-aside>
</template>

<script setup>
import {
  HomeFilled,
  Key,
  OfficeBuilding,
  UploadFilled,
  List,
  User,
  Aim,
  DocumentAdd,
  Reading,
  Help,
} from "@element-plus/icons-vue";

import { onMounted, ref } from "vue";
import axios from "axios";
import { useRoute } from "vue-router";
import { useUserStore } from "../../store/useUserStore";

const route = useRoute();

onMounted(() => {
  getList();
});
const datalist = ref([]);
const getList = async () => {
  var res = await axios.get("/adminapi/rights");
  // console.log(res.data)
  datalist.value = res.data;
};

//图标映射
const mapIcons = {
  HomeFilled,
  Key,
  OfficeBuilding,
  UploadFilled,
  List,
  User,
  Aim,
  DocumentAdd,
  Reading,
  List,
  Help,
};

const { user } = useUserStore();
const checkAuth = (path) => {
  return user.role.rights.includes(path);
};
</script>

<style scoped>
/* 设置悬停时的背景颜色 */
.el-menu-item:hover {
  background-color: #0c4f9c;
}

/* 设置被选中时的背景颜色 */
.el-menu-item.is-active {
  background-color: #00286d;
  color: white;
}

/* 自定义样式表 */
.el-menu-item {
  background-color: #0c4f9c;
  color: white;
}
.el-menu {
  background-color: #0c4f9c;
  text-decoration-color: white;
}

.ti {
  margin: 20px;
}
.ti1 {
  font-size: 16px;
  font-weight: bold;
}
.el-menu {
  border-right: 0 !important;
}
</style>
