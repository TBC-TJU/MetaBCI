<template>
    <div style="margin: 0;padding:0;">
      <el-container style="height:100vh">
        <SideMenu height="2000px"></SideMenu>
        
        <el-container direction="vertical">
          <TopHeader></TopHeader>
          <el-main style="margin: 0;padding:0;">
            <el-scrollbar>
              <router-view></router-view>
            </el-scrollbar>
          </el-main>
        </el-container>
      </el-container>
    </div>
  
</template>

<script setup>


import SideMenu from '../components/MainBox/SideMenu.vue';
import TopHeader from '../components/MainBox/TopHeader.vue';


</script>

<style scoped>

</style>