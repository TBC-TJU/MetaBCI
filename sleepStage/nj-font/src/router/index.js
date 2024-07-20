
import { createRouter, createWebHistory } from 'vue-router'
import Login from '../views/Login.vue'
import MainBox from '../views/MainBox.vue'
import Register from '../views/Register.vue'
import NotFound from '../views/notfound/NotFound.vue'
import RouteConfig from './config'
import { useRouterStore } from '../store/useRouterStore'
import { useUserStore } from '../store/useUserStore'
//#/login createWebHashHistory
///login createWebHistory
import NProgress from 'nprogress'
import 'nprogress/nprogress.css'

const routes = [
    {
        path: "/login",
        name: "login",
        component: Login
    },
    {
        path: "/mainbox",
        name: "mainbox",
        component: MainBox
    },
    {
        path: "/register",
        name: "register",
        component: Register
    }
]



const router = createRouter({
    history: createWebHistory(),//#/login
    routes
})

//添加路由
// router.addRoute("mainbox",{
//     path:"/index",
//     component:Home
// })


//路由拦截
router.beforeEach((to, from, next) => {

    //显示进度条
    NProgress.start();

    const { isGetterRouter } = useRouterStore()
    const {user} = useUserStore()
    // next()
    if (to.name === "login") {
        next()
    } else if(to.name === "register"){
        next()
    }
    else {
        if (!user.role) {
            //跳转登录
            next({
                path: "/login"
            })
        } else {
            if (!isGetterRouter) {
                //remove mainbox
                router.removeRoute("mainbox")
                ConfigRouter()
                next({
                    path:to.fullPath
                })
            }else{
                next()
            }
            
        }
    }

})

router.afterEach(()=>{
    //关闭进度条
    NProgress.done();
})

const ConfigRouter = () => {
    //创建mainbox
    router.addRoute({
        path:"/mainbox",
        name:"mainbox",
        component:MainBox
    })
    let {changeRouter} = useRouterStore()
    RouteConfig.forEach(item => {
        checkPermission(item.path) && router.addRoute("mainbox", item)
    })

    //重定向
    router.addRoute("mainbox",{
        path:"/",
        redirect:"/login"
    })

    //404
    router.addRoute("mainbox",{
        path:"/:pathMatch(.*)*",
        name:"not found",
        component:NotFound
    })
    // console.log(router.getRoutes())

    //true
    changeRouter(true)
}


const checkPermission = (path)=>{
    const {user} = useUserStore()

    return user.role.rights.includes(path)
}
export default router