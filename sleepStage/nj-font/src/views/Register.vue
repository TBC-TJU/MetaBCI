<template>
  <div id="app">
    <vue-particles
      id="tsparticles"
      :particlesInit="particlesInit"
      :particlesLoaded="particlesLoaded"
      :options="config"
    />
  </div>

  <div class="formContainer">
    <div class="main">
      <!-- 整个注册盒子 -->
      <div class="loginbox">
        <!-- 左侧的注册盒子 -->
        <div class="loginbox-in">
          <div class="userbox">
            <el-form ref="ruleFormRef" :model="ruleForm" :rules="rules" status-icon>
              <el-form-item
                label="Neroudoc"
                style="font-size: larger"
                class="formContainer1"
              >
                <!--字体放大居中-->
              </el-form-item>
              <el-form-item
                label="意识障碍睡眠分期和评估"
                class="formContainer2"
              ></el-form-item>
              <el-form-item class="user" label="用户名" prop="username">
                <input v-model="ruleForm.username" style="width: 150px" />
              </el-form-item>
              <el-form-item class="pwdbox" label="密&nbsp;&nbsp;&nbsp;码" prop="password">
                <input v-model="ruleForm.password" type="password" style="width: 150px" />
              </el-form-item>
              <el-form-item class="pwdbox" label="确认密码" prop="password1">
                <input
                  v-model="ruleForm.password1"
                  type="password"
                  style="width: 130px"
                />
              </el-form-item>
              <el-form-item>
                <el-button
                  type="primary"
                  @click="submitForm(ruleFormRef)"
                  class="login_btn"
                  style="margin-left: 50px; width: 150px"
                >
                  注册
                </el-button>
              </el-form-item>
            </el-form>
          </div>
        </div>

        <!-- 右侧的注册盒子 -->
        <div class="background">
          <div class="title">欢迎使用NeuroDoc系统</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useUserStore } from "../store/useUserStore";
import { useRouter } from "vue-router";
import { reactive, ref } from "vue";
import { loadSlim } from "tsparticles-slim";
import { config } from "../util/config";
import { ElMessageBox } from "element-plus";
import axios from "axios";

const particlesInit = async (engine) => {
  //await loadFull(engine);
  await loadSlim(engine);
};

const particlesLoaded = async (container) => {
  console.log("Particles container loaded", container);
};

const ruleFormRef = ref();
const ruleForm = reactive({
  username: "",
  password: "",
});

const rules = reactive({
  username: [{ required: true, message: "请输入用户名", trigger: "blur" }],
  password: [{ required: true, message: "请输入密码", trigger: "blur" }],
  password1: [{ required: true, message: "请重复输入密码", trigger: "blur" }],
});

const { changeUser } = useUserStore();

const router = useRouter();

const submitForm = async (formEl) => {
  if (!formEl) return;
  if (ruleForm.password !== ruleForm.password1) {
    console.log("两次输入密码不一致");
    ruleForm.password = "";
    ruleForm.password1 = "";
    ElMessageBox.alert("两次输入密码不一致");
    return;
  }
  await formEl.validate(async (valid, fields) => {
    if (valid) {
      try {
        console.log("用户名" + ruleForm.username);
        console.log(ruleForm.password);
        console.log(ruleForm);
        var str = new URLSearchParams(ruleForm).toString();
        const response = await axios.post("/adminapi/register", str);
        console.log(response.data);
        if (response.data === "success") {
          ElMessageBox.alert("注册成功", "提示").then(() => {});
          router.push("/login");
        } else {
          ElMessageBox.alert("注册失败", "提示").then(() => {});
        }
      } catch (error) {
        console.error("Error occurred:", error);
      }
    } else {
      console.log("Error in form submission!", fields);
    }
  });
};

const handleLogin1 = () => {
  changeUser({
    id: 1,
    username: "admin",
    password: "123",
    role: {
      roleName: "管理员",
      roleType: 1,
      rights: [
        "/index",
        "/data-upload",
        "/data-analysis",
        "/data-show",
        "/manage",
        "/manage/rightlist",
        "/manage/rolelist",
        "/more",
      ],
    },
  });
  router.push("/login");
};

const handleLogin2 = () => {
  changeUser({
    id: 2,
    username: "kerwin",
    password: "123",
    role: {
      roleName: "用户",
      roleType: 2,
      rights: [
        "/index",
        "/data-upload",
        "/monitor",
        "/data-analysis",
        "/data-show",
        "/more",
        "/data",
        "/data-report",
      ],
    },
  });
  router.push("/login");
};
</script>

<style lang="scss" scoped>
.formContainer {
  width: 500px;
  height: 300px;
  position: fixed;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  color: white;
  text-shadow: 2px 2px 5px black;
  text-align: center;
  display: block;
  z-index: 100;
  .ruleForm {
    margin-top: 50px;
  }

  :deep(.el-form-item__label) {
    color: white;
    font-size: 20px;
    font-weight: 700;
  }
}
.bt {
  width: 150px;
}
.loginbox {
  display: flex;
  position: absolute;
  width: 900px;
  height: 500px;
  border-radius: 20px;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  box-shadow: 0 12px 16px 0 rgba(0, 0, 0, 0.24), 0 17px 50px 0 #4e5265;
}
.loginbox-in {
  background-color: #0c4f9cba;
  width: 300px;
}
.userbox {
  margin-top: 100px;
  height: 30px;
  width: 230px;
  display: flex;
  margin-left: 25px;
}

.formContainer1 {
  :deep(.el-form-item__label) {
    color: white;
    font-size: 40px;
    font-weight: 700;
  }
  margin-left: 20px;
}
.formContainer2 {
  :deep(.el-form-item__label) {
    color: white;
    font-size: 15px;
    font-weight: 700;
  }
  margin-left: 40px;
  margin-top: -20px;
}

.background {
  width: 600px;
  height: auto;
  background-image: url("/public/logo1.png");
  background-size: cover;
  font-family: sans-serif;
}
.title {
  margin-top: 400px;
  font-weight: bold;
  font-size: 30px;
  color: white;
}
.title:hover {
  font-size: 21px;
  transition: all 0.4s ease-in-out;
  cursor: pointer;
}
.uesr-text {
  position: left;
}
input {
  outline-style: none;
  border: 0;
  border-bottom: 1px solid #e9e9e9;
  background-color: transparent;
  height: 20px;
  font-family: sans-serif;
  font-size: 20px;
  color: #a5a6c2;
  font-weight: bold;
}
/* input::-webkit-input-placeholder{
color:#E9E9E9;
} */
input:focus {
  border-bottom: 2px solid rgb(93, 116, 199);
  background-color: transparent;
  transition: all 0.2s ease-in;
  font-family: sans-serif;
  font-size: 20px;
  color: #a5a6c2;
  font-weight: bold;
}
input:hover {
  border-bottom: 2px solid rgb(93, 116, 199);
  background-color: transparent;
  transition: all 0.2s ease-in;
  font-family: sans-serif;
  font-size: 20px;
  color: #a5a6c2;
  font-weight: bold;
}

input:-webkit-autofill {
  /* 修改默认背景框的颜色 */
  box-shadow: 0 0 0px 1000px #89ab9e inset !important;
  /* 修改默认字体的颜色 */
  -webkit-text-fill-color: #445b53;
}

input:-webkit-autofill::first-line {
  /* 修改默认字体的大小 */
  font-size: 15px;
  /* 修改默认字体的样式 */
  font-weight: bold;
}

.login_btn {
  background-color: white; /* Green */
  border: none;
  color: #07083a;
  padding: 5px 22px;
  text-align: center;
  text-decoration: none;
  font-size: 13px;
  border-radius: 20px;
  outline: none;
  margin-top: 20px;
}
.login_btn:hover {
  box-shadow: 0 12px 16px 0 rgba(0, 0, 0, 0.24), 0 17px 50px 0 rgba(0, 0, 0, 0.19);
  cursor: pointer;
  background-color: #e4e6ff;
  color: #07083a;
  transition: all 0.2s ease-in;
}
</style>
