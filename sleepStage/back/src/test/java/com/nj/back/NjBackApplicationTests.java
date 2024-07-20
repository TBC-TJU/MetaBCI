package com.nj.back;

import com.nj.back.dao.RightMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class NjBackApplicationTests {

	@Autowired
	private RightMapper rightMapper;
	@Test
	void contextLoads() {
		System.out.println(rightMapper.getRightList());
	}

}
