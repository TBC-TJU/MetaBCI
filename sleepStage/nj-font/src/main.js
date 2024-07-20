import { createApp } from 'vue'
import App from './App.vue'
import router from './router';
import { createPinia } from 'pinia'
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate'

import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'

import Particles from "vue3-particles";

import ECharts from 'vue-echarts'  // 引入ECharts
import "echarts";
  

const pinia = createPinia()
pinia.use(piniaPluginPersistedstate)

createApp(App).use(router).use(pinia).use(ElementPlus).use(Particles).component('ECharts',ECharts) .mount('#app')

