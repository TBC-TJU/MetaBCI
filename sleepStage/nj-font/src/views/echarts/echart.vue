<template>
  <div>
    <e-charts id="main" class="chart" :option="option" />
  </div>
</template>

<script setup>
import { onMounted } from "vue";
import * as echarts from 'echarts';

onMounted(() => {
 
});

</script>

<style scoped>
.chart {
  width: 800px;
  height: 500px;
}
</style>
