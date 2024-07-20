<template>
  <div class="wrapper" >
    <div class="bg" style="width:800px;padding-top:10px;height: 690px;">
      <div class="ti" style="position: fixed;top:10px;margin-left:10px;color:white;font-size:30px;">数据上传</div>
    <el-form  label-width="120px" ref="dataFormRef" :model="dataForm" :rules="rules" status-icon>
      <el-form-item label="被试名称"  prop="name" class="ti1">
        <el-input v-model="dataForm.name" />
      </el-form-item>
      <el-form-item label="数据类型" prop="type" class="ti1">
        <el-select v-model="dataForm.type" placeholder="请选择数据类型">
          <el-option label="edf" value="edf" />
          <el-option label="csv" value="csv" />
          <el-option label="npz" value="npz" />
          <el-option label="xls" value="xls" />
        </el-select>
      </el-form-item>
  
      <el-form-item label="数据描述" prop="content" class="ti1">
        <el-input v-model="dataForm.content" type="textarea" />
      </el-form-item>
      <el-form-item label="上传文件" prop="upload" class="ti1">
        <el-upload :file-list="fileList" class="upload-demo" drag action="/adminapi/upload" multiple v-model="dataForm.upload" accept="acceptFileType" :before-upload="changeFile"
      :on-change="handleChange" :limit="1" style="width: 100%;">
      <el-icon class="el-icon--upload"><upload-filled /></el-icon>
      <div class="el-upload__text">
        点击此处上传文件
      </div>
      <template #tip>
        <div class="el-upload__tip">
          最大文件上传数为1。
      </div>
      </template>
    </el-upload>
      </el-form-item>
      <el-form-item>
        <el-checkbox label="我已阅读并同意《数据使用说明》" v-model="dataForm.agree" @change="handleAgreeChange" />
  
      </el-form-item>
  
      <el-form-item>
        <el-button type="primary" @click="onSubmit" style="width: 93px;height: 44px;background-color:#0c4f9c;font-weight:bold;">上传</el-button>
        <el-button style="width: 93px;height: 44px;color:#0c4f9c;font-weight:bold;">取消</el-button>
      </el-form-item>
    </el-form>
    
    </div>
    <div class="bg" style="width:600px;padding-top:10px;height: 690px;">
      <div>
        <Calendar
        backgroundText
        class-name="select-mode"
        :remarks="remarks" class="cal" style="width: 400px;"
      />
      </div>
      <div><br><br></div>
      <!--<div style="display: flex;">
        <div style="flex: 1;" class="sl">2023.10.10<br><br>10份</div>
        <div style="flex: 1;" class="sl">2023.10.11<br><br>11份</div>
        <div style="flex: 1;" class="sl">2023.10.12<br><br>12份</div>
      </div>    -->  
    </div>
  </div>
  
</template>

<script lang="ts" setup>
import { h,reactive } from 'vue'
import { ref ,computed} from 'vue'
import { genFileId ,ElMessage,ElMessageBox} from 'element-plus'
import type { UploadInstance, UploadProps, UploadRawFile } from 'element-plus'
import axios from 'axios'
import { UploadFilled } from '@element-plus/icons-vue'
import Calendar from 'mpvue-calendar'

const value = ref(new Date())

const upload = ref<UploadInstance>()

const handleExceed: UploadProps['onExceed'] = (files) => {
  upload.value!.clearFiles()
  const file = files[0] as UploadRawFile
  file.uid = genFileId()
  upload.value!.handleStart(file)
}

const submitUpload = () => {
  upload.value!.submit()
}
const open = () => {
  ElMessageBox({
    title: '提示',
    message: h('p', null, [
      h('span', null, '数据已上传成功!')
    ]),
  })
}

const handleConfirm = () => {
  dataFormRef.value.validate(async (valid, fields) => {
      if (valid) {
          await axios.put(`/adminapi/file`,dataForm)
      } else {
          console.log('error submit!', fields)
      }
  })
}

const onSubmit = () => {
  // 验证所有的表单数据
  if (!dataForm.name || !dataForm.type || !dataForm.content || !dataForm.upload || !dataForm.agree) {
    ElMessage.error('请完整填写所有表单数据并同意《数据使用说明》');
    console.log(dataForm);
    return;
  }
  // 如果所有数据都存在，执行上传操作
  console.log(dataForm);
//  upload.value!.submit();
  handleConfirm();
  open();
  // 这里可以添加额外的操作或重置表单等
};


const dataFormRef = ref()
const dataForm = reactive({
  name: "",
  type: "",
  content: "",
  upload:"",
  agree:""
})

const acceptFileType = computed(() => {
  // 根据 dataForm.type 的值来动态设置 accept 属性
  switch (dataForm.type) {
    case 'edf':
      return 'edf';
    case 'csv':
      return '.csv';
    case 'npz':
      return '.npz';
    case 'xls':
      return '.xls';
    default:
      return ''; // 没有匹配的情况下可以设置为空字符串或其他默认值
  }
});

const rules = reactive({
  name: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
  ],
  type: [
    { required: true, message: '例如EEG', trigger: 'blur' },
  ],
  content: [
    { required: true, message: '请输入相关描述', trigger: 'blur' },
  ],
  file:[
    { required: true, message: '请上传相关文件', trigger: 'blur' },
  ]
})

const fileList = ref<UploadRawFile[]>([]);
const handleChange = (file, fileList) => {
  const fullFilePath = file.name; // 获取完整文件路径
        const fileName = fullFilePath.split('\\').pop(); // 获取文件名
        dataForm.upload = fileName; // 仅保留文件名
    //限制文件数量，此处设置限制1条
    if (fileList.length > 1) {
        fileList.splice(0, 1);
    }
};
const changeFile = (file, fileList) => {
    console.log('file', file, fileList);
    if (!acceptFileType.value) {
        ElMessage.error('请选择支持的文件类型');//限制文件类型
        return false;
    } else {
        if (!file.name.endsWith(acceptFileType.value)) {
            ElMessage.error('只能上传' + acceptFileType.value + '格式的文件');//限制文件类型
            return false;
        }
        
        let param = new FormData();
        param.append('file', file);
        console.log(param);
        
      /*  axios({
            method: 'POST',
            url: 'https://run.mocky.io/v3/9d059bf9-4660-45f2-925d-ce80ad6c4d15',
            data: param
        }).then((response) => {
            console.log(response); //查看接口返回的数据
        }).catch((error) => {
            console.error("错误信息：" + error);
        });*/
    }
};

const handleAgreeChange = (value) => {
  // value 参数表示复选框的新状态，true 表示勾选，false 表示取消勾选
  // 您可以在这里执行任何与用户同意操作相关的逻辑
  
  console.log('用户同意状态:', value);
};

const remarks = ref({'2021-1-13': 'some tings'})

</script>

<style>
.wrapper {
  display: flex;
}
.bg{
  background-color: #f6f6f6;
  width: auto;
  height: 100%;
}
.ti{
  font-size: 48px;
  font-weight: bold;
  padding: 5px;
}
.ti1{
  font-size: 16px;
  font-weight: bold;
  
}
.cal{
  margin-left:50px;

}
.sl{
  width: 90px;
  height: 120px;
  background-color: #ADAED8;
  margin: 15px;
  margin-top: 30px;
  justify-content: center; 
  align-items: center;
  display: flex;
  border-radius: 10px;
  font-size: 18px;
}
</style>
