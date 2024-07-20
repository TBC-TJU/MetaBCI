<template>
  <div>
    <e-charts id="main4" class="chart3" :option="option3" />
  </div>
</template>

<script setup>
import { onMounted } from "vue";
import * as echarts from 'echarts';

onMounted(() => {
  var chartDom = document.getElementById('main4');
  var myChart = echarts.init(chartDom);
  var option3;

  option3 = {
    legend: {
      top: 'bottom'
    },
    toolbox: {
      show: true,
      feature: {
        mark: { show: true },
        dataView: { show: true, readOnly: false },
        restore: { show: true },
        saveAsImage: { show: true }
      }
    },
    series: [
      {
        name: 'Nightingale Chart',
        type: 'pie',
        radius: [20, 100],
        center: ['50%', '50%'],
        roseType: 'area',
        itemStyle: {
          borderRadius: 8
        },
        data: [
          { value: 40, name: 'W期' },
          { value: 38, name: 'N1期' },
          { value: 32, name: 'N2期' },
          { value: 30, name: 'N3期' },
          { value: 28, name: 'REM期' }
        ]
      }
    ]
  };
  option && myChart.setOption(option);
});

</script>

<style scoped>
.chart3 {
  width: 500px;
  height: 300px;
}
</style>
