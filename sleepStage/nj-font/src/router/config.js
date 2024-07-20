import Home from "../views/home/Home.vue";
import DataUpload from "../views/data-upload/DataUpload.vue";
import DataAnalysis from "../views/data-analysis/DataAnalysis.vue";
import DataShow from "../views/data-show/DataShow.vue";
import More from "../views/more/More.vue";
import Manage from '../views/manage/Manage.vue';
import Role from '../views/manage/Role.vue';
import Monitor from '../views/monitor/Monitor.vue'
import Data from '../views/data-history/Data.vue';
import DataReport from '../views/data-history/DataReport.vue';
import Register from '../views/Register.vue';

const routes = [
  {
    path:"/index",
    component:Home
  },
  {
    path:"/data-upload",
    component:DataUpload
  },
  {
    path:"/data-analysis",
    component:DataAnalysis
  },
  {
    path:"/data-show",
    component:DataShow
  },
  {
    path:"/more",
    component:More
  },
  {
    path:"/manage/rightlist",
    component:Manage
  },
  {
    path:"/manage/rolelist",
    component:Role
  },
  {
    path:"/monitor",
    component:Monitor
  },
  {
    path:"/data",
    component:Data
  },
  {
    path:"/data-report",
    component:DataReport
  },
]

export default routes;