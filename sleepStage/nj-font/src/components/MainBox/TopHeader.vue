<template>
  <div>
    <el-header>
      <div style="float: right; display: block; margin-top: 15px">
        <el-dropdown>
          <el-avatar :size="40" :src="circleUrl" />
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item>{{ user.role.roleName }}</el-dropdown-item>
              <el-dropdown-item @click="handleExit">退出</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
        <span style="line-height: 40px">&nbsp;&nbsp;{{ user.username }}</span
        >&nbsp;&nbsp;&nbsp;
      </div>
    </el-header>
  </div>
</template>

<script setup>
import { useUserStore } from "../../store/useUserStore";
import { useRouter } from "vue-router";
import { useRouterStore } from "../../store/useRouterStore";

const { changeUser, user } = useUserStore();
const { changeRouter } = useRouterStore();
const router = useRouter();

const circleUrl = "/avatar.png";

const handleExit = () => {
  changeUser({});

  changeRouter(false);
  router.push("/login");
};
</script>

<style lang="scss" scoped>
.el-header {
  background-color: #518aca;
  color: white;
  height: 80px;
  margin: 0px;
  padding: 0px;
}
</style>
