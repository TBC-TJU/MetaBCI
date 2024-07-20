package com.nj.back.dao;

import com.nj.back.pojo.History;
import org.apache.ibatis.annotations.*;

@Mapper
public interface HistoryDao {
    @Insert("INSERT INTO history(date,time,sleeptime,wtime,n1time,n2time,n3time,remtime,wrate,n1rate,n2rate,n3rate,remrate,type,upload,id)" +
            " VALUES (#{date}, #{time}, #{sleeptime}, #{wtime}, #{n1time}, #{n2time}, #{n3time}, #{remtime}, #{wrate}, #{n1rate}, #{n2rate}, #{n3rate}, #{remrate}, #{type},#{upload},#{id})")

    void insertHistory(History history);
    @Update("UPDATE file SET tag = 'YES' WHERE upload = #{filename}")
    void updateFileTag(@Param("filename") String filename);
    @Select("SELECT COUNT(*) FROM history WHERE date = #{dateValue}")
    int countByDate(@Param("dateValue") String dateValue);
    @Select("SELECT id FROM file WHERE upload = #{fileValue}")
    int countByFilename(@Param("fileValue") String fileValue);
}
