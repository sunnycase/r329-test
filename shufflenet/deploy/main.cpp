
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include "label.h"


#include "standard_api.h"
#include <iostream>
#include <sys/time.h>

int label_oft = 1;

static int cal_interval(struct timeval start, struct timeval end)
{
    struct timeval interval;
    if (end.tv_usec >= start.tv_usec) {
        interval.tv_usec = end.tv_usec - start.tv_usec;
        interval.tv_sec = end.tv_sec - start.tv_sec;
    } else  {
        interval.tv_usec = 1000000 + end.tv_usec - start.tv_usec;
        interval.tv_sec = end.tv_sec - 1 - start.tv_sec;
    }
    return (int)(interval.tv_sec*1000 + interval.tv_usec/1000);
}

#define T_START() {gettimeofday(&start, NULL);}
#define T_END(x)  {gettimeofday(&end, NULL); \
                int tmp=cal_interval(start, end); \
                printf("Step %s cost %d ms\r\n", x, tmp);\
                }

typedef struct {
    int index;
    int8_t val;
} int8_data_t;

typedef struct {
    int index;
    uint8_t val;
} uint8_data_t;

int uint8_comp_down(const void*p1, const void*p2) {
	//<0, 元素1排在元素2之前；即降序
	int tmp = (((uint8_data_t*)p2)->val) - (((uint8_data_t*)p1)->val);
	return tmp;  
}

int int8_comp_down(const void*p1, const void*p2) {
	//<0, 元素1排在元素2之前；即降序
	int tmp = (((int8_data_t*)p2)->val) - (((int8_data_t*)p1)->val);
	return tmp;  
}


static void decode_result_int8(int8_t *result, uint32_t size)
{
    int8_data_t* buf = (int8_data_t*)malloc(sizeof(int8_data_t)*size);
    if(buf == NULL) return;
    for(int i=0; i < size; i++) {
        buf[i].index = i;
        buf[i].val = result[i];
    }
    qsort(buf, size, sizeof(int8_data_t), int8_comp_down);
    printf("Decode Result:\r\n");
    for(int i=0; i < 5; i++) {
        printf("    %d: class %4d, prob %3d; label: %s\r\n", i, buf[i].index, buf[i].val, labels[buf[i].index-label_oft]);
    }
    free(buf);
    return;
}

static void decode_result_uint8(uint8_t *result, uint32_t size)
{
    uint8_data_t* buf = (uint8_data_t*)malloc(sizeof(uint8_data_t)*size);
    if(buf == NULL) return;
    for(int i=0; i < size; i++) {
        buf[i].index = i;
        buf[i].val = result[i];
    }
    qsort(buf, size, sizeof(uint8_data_t), uint8_comp_down);
    printf("Decode Result:\r\n");
    for(int i=0; i < 5; i++) {
        printf("    %d: class %4d, prob %3u; label: %s\r\n", i, buf[i].index, buf[i].val, labels[buf[i].index-label_oft]);
    }
    free(buf);
    return;
}

int loadFromBMP(const char* bmpPath, int w, int h, void* buffer)
{
	FILE* fp = fopen(bmpPath, "rb");
	if (fp == NULL)
	{
		printf("fopen %s failed\n", bmpPath);
		return -1;
	}
    uint8_t* bmpbuf = (uint8_t*)malloc(w*h*3+100); 
    if(bmpbuf == NULL)
    {
        printf("loadbmp buf err\r\n");
        return -1;
    }
    
	int nread = fread(bmpbuf, 1, w*h*3+54, fp);
	if (nread != w*h*3+54)
	{
		printf("fread bmp failed %d\n", nread);
		return -1;
	}
	fclose(fp);
	//整理顺序
	int8_t* buf = (int8_t*)buffer;
	char* bmp = (char*)(bmpbuf+54); //像素数据在第54字节开始
	for(int y=0; y<h; y++) {
		for(int x=0; x<w; x++) {
			buf[3*(w*(h-1-y)+x)+0] = (bmp[3*(w*y+x)+2]); //BGR->RGB
			buf[3*(w*(h-1-y)+x)+1] = (bmp[3*(w*y+x)+1]);
			buf[3*(w*(h-1-y)+x)+2] = (bmp[3*(w*y+x)+0]);
		}
	}
    
    free(bmpbuf);
	return 0;
}

int loadFromBIN(const char* binPath, int w, int h, void* buffer)
{
	FILE* fp = fopen(binPath, "rb");
	if (fp == NULL)
	{
		printf("fopen %s failed\n", binPath);
		return -1;
	}
    
	int nread = fread(buffer, 1, w*h*3, fp);
	if (nread != w*h*3)
	{
		printf("fread bin failed %d\n", nread);
		return -1;
	}
	fclose(fp);
	return 0;
}

int main(int argc, char *argv[])
{
    int ret = 0;
    uint32_t job_id=0;
    int32_t time_out=-1;
    bool finish_job_successfully = true;
    int model_inw, model_inh, model_inch, model_outw, model_outh, model_outch, img_size;    
    int8_t* bmpbuf;
    struct timeval start, end, interval;
    
    printf("Zhouyi ShuffleNet test program: \r\n");
    printf("Usage: \r\n");
    printf("    ./shufflenet aipu.bin input.bmp\r\n");
    printf("    NOTE: bmp file width/height must correct\r\n");
    
    aipu_ctx_handle_t * ctx = NULL;
    aipu_status_t status = AIPU_STATUS_SUCCESS;
    const char* status_msg =NULL;
    aipu_graph_desc_t gdesc;
    aipu_buffer_alloc_info_t info;

    //Step 0: parse input argv
	if(argc < 3) {
		printf("argc=%d error\r\n", argc);
        return -1;
	}
    char* file_model= argv[1];
	char* file_bmp	= argv[2];
    size_t file_bmp_len = strlen(file_bmp);
    int signed_flag = 1;
    
    
    //Step1: init ctx handle
    T_START();
    status = AIPU_init_ctx(&ctx);       
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_init_ctx: %s\n", status_msg);
        ret = -1;
        goto out;
    }
    T_END("1 init ctx handle");

    //Step2: load graph
    T_START();
    status = AIPU_load_graph_helper(ctx, file_model, &gdesc);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_load_graph_helper: %s\n", status_msg);
        ret = -1;
        goto deinit_ctx;
    }
    printf("[DEMO INFO] AIPU load graph successfully.\n");
    T_END("2 load graph");

    //Step3: alloc tensor buffers
    T_START();
    status = AIPU_alloc_tensor_buffers(ctx, &gdesc, &info);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_alloc_tensor_buffers: %s\n", status_msg);
        ret = -1;
        goto unload_graph;
    }
    T_END("3 alloc tensor buffers");

    //Step4: read pic   
    T_START();
    model_inw  = gdesc.inputs.desc[0].fmt.shape.W;
    model_inh  = gdesc.inputs.desc[0].fmt.shape.H;
    model_inch = gdesc.inputs.desc[0].fmt.shape.C;
    printf("    Model input:  W=%3d, H=%3d, C =%d, size=%d\r\n", model_inw, model_inh, model_inch, info.inputs.tensors[0].size);
    img_size = model_inw*model_inh*model_inch;
    bmpbuf = (int8_t*)malloc(img_size);
    if(bmpbuf == NULL) {
        printf("[DEMO ERROR] bmpbuf alloc error\r\n");
        goto free_tensor_buffers;
    }
    if(strcmp(file_bmp + file_bmp_len - 4, ".bmp") == 0) {
        printf("loadFromBMP: %s\r\n", file_bmp);
        ret = loadFromBMP(file_bmp, model_inw, model_inh, bmpbuf);
        if(ret != 0) {
            printf("[DEMO ERROR] loadFromBMP error\r\n");
            goto free_bmpbuf;
        }
        for(int i=0; i < img_size; i++) {    // uint8_t -> int8_t
            bmpbuf[i] -= 127;
        }
    } else {
        printf("loadFromBIN: %s\r\n", file_bmp);
        ret = loadFromBIN(file_bmp, model_inw, model_inh, bmpbuf);
        if(ret != 0) {
            printf("[DEMO ERROR] loadFromBIN error\r\n");
            goto free_bmpbuf;
        }
    }
    T_END("4 read pic");

    //Step5: call pic 
    T_START();
    memcpy(info.inputs.tensors[0].va, bmpbuf, info.inputs.tensors[0].size);
    status = AIPU_create_job(ctx, &gdesc, info.handle, &job_id);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_create_job: %s\n", status_msg);
        ret = -1;
        goto free_bmpbuf;
    }
    T_END("5 AIPU_create_job");
    
    T_START();
    status = AIPU_finish_job(ctx, job_id, time_out);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_finish_job: %s\n", status_msg);
        finish_job_successfully = false;
    } else {
        finish_job_successfully = true;
    }
    T_END("5 AIPU_finish_job");
    T_START();
    if (finish_job_successfully) {
        int8_t *result = (int8_t *)info.outputs.tensors[0].va;
        uint32_t size = info.outputs.tensors[0].size;
        if(signed_flag == 0) {
            decode_result_uint8((uint8_t*)result, size);
        } else {
            decode_result_int8(result, size);
        }
        
    }
    T_END("5 decode_result");
    T_START();
    status = AIPU_clean_job(ctx, job_id);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[TEST ERROR] AIPU_clean_job: %s\n", status_msg);
        ret = -1;
        goto free_bmpbuf;
    }
    T_END("5 AIPU_clean_job");


free_bmpbuf:
    free(bmpbuf);

free_tensor_buffers:
    status = AIPU_free_tensor_buffers(ctx, info.handle);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_free_tensor_buffers: %s\n", status_msg);
        ret = -1;
    }
unload_graph:
    status = AIPU_unload_graph(ctx, &gdesc);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_unload_graph: %s\n", status_msg);
        ret = -1;
    }
deinit_ctx:
    status = AIPU_deinit_ctx(ctx);
    if (status != AIPU_STATUS_SUCCESS) {
        AIPU_get_status_msg(status, &status_msg);
        printf("[DEMO ERROR] AIPU_deinit_ctx: %s\n", status_msg);
        ret = -1;
    }

out:

    return ret;
}
