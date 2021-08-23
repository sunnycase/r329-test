/**********************************************************************************
 * This file is CONFIDENTIAL and any use by you is subject to the terms of the
 * agreement between you and Arm China or the terms of the agreement between you
 * and the party authorised by Arm China to disclose this file to you.
 * The confidential and proprietary information contained in this file
 * may only be used by a person authorised under and to the extent permitted
 * by a subsisting licensing agreement from Arm China.
 *
 *        (C) Copyright 2020 Arm Technology (China) Co. Ltd.
 *                    All rights reserved.
 *
 * This entire notice must be reproduced on all copies of this file and copies of
 * this file may only be made by a person if such person is permitted to do so
 * under the terms of a subsisting license agreement from Arm China.
 *
 *********************************************************************************/

/**
 * @file  low_level_api_v0.0.1.h
 * @brief AIPU User Mode Driver (UMD) Low Level API header
 * @version 0.0.1
 */

#ifndef _LOW_LEVEL_API_V_0_0_1_H_
#define _LOW_LEVEL_API_V_0_0_1_H_

#include <stdint.h>

typedef int32_t AIPU_HANDLE;

typedef struct aipu_buffer {
    uint32_t pa;
    void* va;
    uint32_t size;
} aipu_buffer_t;

typedef enum {
    AIPU_LL_STATUS_SUCCESS = 0x0,
    AIPU_LL_STATUS_ERROR_OPEN_FAILED,
    AIPU_LL_STATUS_ERROR_INVALID_PTR,
    AIPU_LL_STATUS_ERROR_INVALID_HANDLE,
    AIPU_LL_STATUS_ERROR_INVALID_OFFSET,
    AIPU_LL_STATUS_ERROR_INVALID_OP,
    AIPU_LL_STATUS_ERROR_INVALID_SIZE,
    AIPU_LL_STATUS_ERROR_INVALID_PA,
    AIPU_LL_STATUS_ERROR_READ_REG_FAIL,
    AIPU_LL_STATUS_ERROR_WRITE_REG_FAIL,
    AIPU_LL_STATUS_ERROR_MALLOC_FAIL,
    AIPU_LL_STATUS_ERROR_FREE_FAIL
} aipu_ll_status_t;

/**
 * @brief This API is used to open an AIPU device.
 *
 * @param handle Pointer to a memory location allocated by application where UMD stores the
 *               device handle for further operations to use
 *
 * @retval AIPU_LL_STATUS_SUCCESS
 * @retval AIPU_LL_STATUS_ERROR_OPEN_FAILED
 * @retval AIPU_LL_STATUS_ERROR_INVALID_PTR
 */
aipu_ll_status_t AIPU_LL_open(AIPU_HANDLE* handle);
/**
 * @brief This API is used to close an opened AIPU device.
 *
 * @param handle Device handle returned by AIPU_LL_open
 *
 * @retval AIPU_LL_STATUS_SUCCESS
 * @retval AIPU_LL_STATUS_ERROR_INVALID_HANDLE
 * @retval AIPU_LL_STATUS_ERROR_INVALID_PTR
 */
aipu_ll_status_t AIPU_LL_close(AIPU_HANDLE handle);
/**
 * @brief This API is used to get the version information of AIPU device.
 *
 * @param handle  Device handle returned by AIPU_LL_open
 * @param version Pointer to a memory location allocated by application where UMD stores the
 *                AIPU version number
 *
 * @retval AIPU_LL_STATUS_SUCCESS
 * @retval AIPU_LL_STATUS_ERROR_INVALID_HANDLE
 */
aipu_ll_status_t AIPU_LL_get_hw_version(AIPU_HANDLE handle, uint32_t* version);
/**
 * @brief This API is used to capture value of an AIPU external register.
 *
 * @param handle Device handle returned by AIPU_LL_open
 * @param offset Register offset
 * @param value  Pointer to a memory location allocated by application where UMD stores the
 *               register readback value
 *
 * @retval AIPU_LL_STATUS_SUCCESS
 * @retval AIPU_LL_STATUS_ERROR_INVALID_HANDLE
 * @retval AIPU_LL_STATUS_ERROR_INVALID_PTR
 * @retval AIPU_LL_STATUS_ERROR_INVALID_OFFSET
 * @retval AIPU_LL_STATUS_ERROR_READ_REG_FAIL
 */
aipu_ll_status_t AIPU_LL_read_reg32(AIPU_HANDLE handle, uint32_t offset, uint32_t* value);
/**
 * @brief This API is used to write an AIPU external register.
 *
 * @param handle Device handle returned by AIPU_LL_open
 * @param offset Register offset
 * @param value  Value to be write into this register
 *
 * @retval AIPU_LL_STATUS_SUCCESS
 * @retval AIPU_LL_STATUS_ERROR_INVALID_HANDLE
 * @retval AIPU_LL_STATUS_ERROR_INVALID_OFFSET
 * @retval AIPU_LL_STATUS_ERROR_INVALID_OP
 * @retval AIPU_LL_STATUS_ERROR_WRITE_REG_FAIL
 */
aipu_ll_status_t AIPU_LL_write_reg32(AIPU_HANDLE handle, uint32_t offset, uint32_t value);
/**
 * @brief This API is used to request to allocate a host-device shared physically contiguous buffer.
 *
 * @param handle Device handle returned by AIPU_LL_open
 * @param size   Buffer size requested
 * @param align  Address alignment in page (4KB by default)
 * @param buf    Pointer to a memory location allocated by application where UMD stores the
 *               successfully allocated buffer info.
 *
 * @retval AIPU_LL_STATUS_SUCCESS
 * @retval AIPU_LL_STATUS_ERROR_INVALID_HANDLE
 * @retval AIPU_LL_STATUS_ERROR_INVALID_PTR
 * @retval AIPU_LL_STATUS_ERROR_INVALID_SIZE
 * @retval AIPU_LL_STATUS_ERROR_MALLOC_FAIL
 */
aipu_ll_status_t AIPU_LL_malloc(AIPU_HANDLE handle, uint32_t size, uint32_t align, aipu_buffer_t* buf);
/**
 * @brief This API is used to request to free a buffer allocated by AIPU_LL_malloc.
 *
 * @param handle Device handle returned by AIPU_LL_open
 * @param buf    Buffer descriptor pointer returned by AIPU_LL_malloc
 *
 * @retval AIPU_LL_STATUS_SUCCESS
 * @retval AIPU_LL_STATUS_ERROR_INVALID_HANDLE
 * @retval AIPU_LL_STATUS_ERROR_INVALID_PA
 * @retval AIPU_LL_STATUS_ERROR_FREE_FAIL
 */
aipu_ll_status_t AIPU_LL_free(AIPU_HANDLE handle, aipu_buffer_t* buf);

#endif /* _LOW_LEVEL_API_V_0_0_1_H_ */
