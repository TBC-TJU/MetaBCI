<template>
  <div>
    <e-charts id="main3" class="chart1" :option="option2" />
  </div>
</template>

<script setup>
import { onMounted } from "vue";
import * as echarts from 'echarts';

onMounted(() => {
  
var chartDom = document.getElementById('main3');
var myChart = echarts.init(chartDom);
var option2;

let colors = ['#e4e6ff','#9fa3f7','#C4C6F5','#E3C4F7','#8CA6E2']
option2 = {
    xAxis: {
        type: 'category',
        data: ['W', 'N1', 'N2', 'N3', 'REM']
    },
    yAxis: {
        type: 'value'
    },
    series: [{
        color: colors,
        data: [0.6,4,2,3,7],
        type: 'bar',
        itemStyle: {
            normal: {
                    color: function(params) {
                  let idx = params.dataIndex
                  return colors[idx]
                },
                barBorderRadius: [50, 50, 0, 0], // 左上，右上，右下，左下
            }
        }    
    }]
};



});

</script>

<style scoped>
.chart1 {
  width: 500px;
  height: 300px;
}
</style>
