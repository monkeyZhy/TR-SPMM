#include <mma.h>
#include <cstdint>
#include <stdio.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

        struct mmaComputeUtils_tf32_v2{
        // Shared memory buffers
        // const uint32_t* lhs_tile_;
        uint32_t* rhs_fragment;
        // Register file fragment to accumulate results into
        float * output_fragment_;
        int lane_id_;
        uint32_t *lhs_fragment;

        // Constructor
        __device__ __forceinline__ mmaComputeUtils_tf32_v2(
            float* dense_tile,
            float* output_fragment,
            int lane_id,
            float *sparse_fragment):
            // lhs_tile_(reinterpret_cast<const  uint32_t*>(lhs_tile)),
            lane_id_(lane_id),
            rhs_fragment(reinterpret_cast<uint32_t*>(dense_tile)),
            output_fragment_(output_fragment),
            lhs_fragment(reinterpret_cast<uint32_t*>(sparse_fragment)){}
        
        // Compute
        __device__ __forceinline__ void TileMAC(){

        // float rhs_fragment1[2];
        // // densetile + 所在的warp + 第1or2个32块。符合mma数据布局
        // #pragma unroll
        // for(int i=0;i<2;i++){
        //     rhs_fragment1[i] =  *(dense_tile_ + (lane_id_/32)*64 + lane_id_%32 + i*32); 
        // }
        // uint32_t* rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment1);
    
        asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(lhs_fragment[0]), "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));
            
        }

        __device__ __forceinline__ void TileMACResidue(){
        // // uint32_t lhs_fragment[1];
        // float rhs_fragment1[2];
        // // densetile + 所在的warp + 第1or2个32块。符合mma数据布局
        // #pragma unroll
        // for(int i=0;i<2;i++){
        //     rhs_fragment1[i] = *(dense_tile_ + (lane_id_/32)*64 + lane_id_%32 + i*32); 
        // }
        // uint32_t* rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment1);
            

       asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(lhs_fragment[0]), "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));      
        }
        
        
    };
//仅用于CUDA计算
struct cudaComputeUtils_tf32{
    const float * c_rhs_matrix_;
    const int *c_col_indices_;
    const float *c_values_;
    float * output_matrix_;
    int nOri_;
    int mOri_;
    int warpin_id_;
    // int warp_id_;
    // const int *c_row_offsets_;
    // Constructor
    __device__ __forceinline__ cudaComputeUtils_tf32(
        const float * rhs_matrix,
        const int *c_col_indices,
        const float *c_values,
        float * output_matrix,
        int nOri,
        int mOri,
        int warpin_id
        // int warp_id,
        // const int * c_row_offsets
    ):
    c_rhs_matrix_(rhs_matrix),
    c_col_indices_(c_col_indices),
    c_values_(c_values),
    output_matrix_(output_matrix),
    nOri_(nOri),
    mOri_(mOri),
    warpin_id_(warpin_id)
    // warp_id_(warp_id),
    // c_row_offsets_(c_row_offsets)
    {}
    
    // CUDA Compute
    __device__ __forceinline__ void cudaCompute( int nonzeros, int dimN_index, int row, int atomic){

        // for(int k=0;k<2;k++)
        // {
        //     float res = 0.0;
        //     int c_col_offset = dimN_index + k*32 + warpin_id_;
        //     if(c_col_offset<nOri_)
        //     { 
        //         #pragma unroll
        //         for(int i=0; i<nonzeros; i++)
        //         {
        //             int col = __ldg(c_col_indices_ + i);
        //             float b = __ldg(c_rhs_matrix_ + col*nOri_ + c_col_offset);  
        //             res +=  __ldg(c_values_ + i) * b;
        //         }
        //         if(atomic==0)
        //         *(output_matrix_ + row*nOri_ + c_col_offset) = res;
        //         else  atomicAdd((output_matrix_ + row*nOri_ + c_col_offset) , res);
        //     }
        // }  
        float res[2] = {0.0, 0.0};
        // int ites = (nOri_ - dimN_index - warpin_id_) / 32;
        #pragma unroll
        for(int i=0; i<nonzeros; i++)
        {
            float a =  __ldg(c_values_ + i);
            int col = __ldg(c_col_indices_ + i);
            for(int k=0;k<2;k++)
            {
                int c_col_offset = dimN_index + k*32 + warpin_id_;
                if(c_col_offset<nOri_)
                {          
                    float b = __ldg(c_rhs_matrix_ + col*nOri_ + c_col_offset);    
                    res[k] +=  __ldg(c_values_ + i) * b;
                }
                if(i==nonzeros-1)
                {
                    if(atomic==0)
                    *(output_matrix_ + row*nOri_ + c_col_offset) = res[k];
                    else  atomicAdd((output_matrix_ + row*nOri_ + c_col_offset) , res[k]);
                }
            }
        }
        
          
    }
};



struct cudaComputeUtils_tf32_v3{
    const float * c_rhs_matrix_;
    const int *c_col_indices_;
    const float *c_values_;
    float * output_matrix_;
    int nOri_;
    int mOri_;
    int warpin_id_;
    float * c_part_result_;
    // int warp_id_;
    // const int *c_row_offsets_;
    // Constructor
    __device__ __forceinline__ cudaComputeUtils_tf32_v3(
        const float * rhs_matrix,
        const int *c_col_indices,
        const float *c_values,
        float * output_matrix,
        int nOri,
        int mOri,
        int warpin_id,
        float *c_part_result
        // int warp_id,
        // const int * c_row_offsets
    ):
    c_rhs_matrix_(rhs_matrix),
    c_col_indices_(c_col_indices),
    c_values_(c_values),
    output_matrix_(output_matrix),
    nOri_(nOri),
    mOri_(mOri),
    warpin_id_(warpin_id),
    c_part_result_(c_part_result)
    // warp_id_(warp_id),
    // c_row_offsets_(c_row_offsets)
    {}
    
    // CUDA Compute
    __device__ __forceinline__ void cudaCompute( int nonzeros, int row, int atomic, int iteras){
            // int cur = 0;
        #pragma unroll
        for(int i=0; i<nonzeros; i++)
        {
            float a =  __ldg(c_values_ + i);
            int col = __ldg(c_col_indices_ + i);
            // c_rhs_matrix_ += col*nOri_;
            for(int k=0;k<iteras;k++)
            {
                int c_col_offset = k*32 + warpin_id_;
                if(c_col_offset<nOri_)
                {          
                    float b = __ldg(c_rhs_matrix_ + col*nOri_ + c_col_offset);    
                    *(c_part_result_ + c_col_offset) +=  a * b;
                    // if(threadIdx.x==0 and row == 8)
                    // printf("%f %f %f %d\n",*(c_part_result_ + c_col_offset), a ,b, cur++);
                    if(i==nonzeros-1)
                    {
                        if(atomic==0)
                        *(output_matrix_ + row*nOri_ + c_col_offset) = *(c_part_result_ + c_col_offset);
                        else  atomicAdd((output_matrix_ + row*nOri_ + c_col_offset) , *(c_part_result_ + c_col_offset));
                    }
                }
            }
        }
        
          
    }
};

//仅用于tcu&CUDA中的cuda计算，
struct cudaComputeUtils_tf32_shared{
    const float * c_rhs_matrix_;
    const int *c_col_indices_;
    const float *c_values_;
    float * output_matrix_;
    int nOri_;
    int mOri_;
    int warpin_id_;
    int warp_id_;
    const int *c_row_offsets_;
    // Constructor
    __device__ __forceinline__ cudaComputeUtils_tf32_shared(
        const float * rhs_matrix,
        const int *c_col_indices,
        const float *c_values,
        float * output_matrix,
        int nOri,
        int mOri,
        int warpin_id
    ):
    c_rhs_matrix_(rhs_matrix),
    c_col_indices_(c_col_indices),
    c_values_(c_values),
    output_matrix_(output_matrix),
    nOri_(nOri),
    mOri_(mOri),
    warpin_id_(warpin_id)
    {}
    
    // CUDA Compute
    __device__ __forceinline__ void cudaCompute(int nonzeros, int dimN_index){
        for(int k=0;k<2;k++)
        {
            float res = 0.0;
            int c_col_offset = dimN_index + k*32 + warpin_id_;
            if(c_col_offset<nOri_)
            { 
                #pragma unroll
                for(int i=0; i<nonzeros; i++)
                {
                    int col = __ldg(c_col_indices_ + i);
                    float b = __ldg(c_rhs_matrix_ + col*nOri_ + c_col_offset);  
                    res +=  __ldg(c_values_ + i) * b;
                }
                //把res写回global mempry
                // *(c_output_fragment_ + warp_id_*64 + k*32 + warpin_id_) = res;
                atomicAdd((output_matrix_ + c_col_offset),res);
            }
        }  
    }
};

struct mmaComputeUtils_tf32{
    // Shared memory buffers
    // const uint32_t* lhs_tile_;
    uint32_t* rhs_fragment;
    // Register file fragment to accumulate results into
    float * output_fragment_;
    int lane_id_;
    uint32_t *lhs_fragment;

    // Constructor
    __device__ __forceinline__ mmaComputeUtils_tf32(
        float* dense_tile,
        float* output_fragment,
        int lane_id,
        float *sparse_fragment):
        // lhs_tile_(reinterpret_cast<const  uint32_t*>(lhs_tile)),
        lane_id_(lane_id),
        rhs_fragment(reinterpret_cast<uint32_t*>(dense_tile)),
        output_fragment_(output_fragment),
        lhs_fragment(reinterpret_cast<uint32_t*>(sparse_fragment)){}
    
    // Compute
    __device__ __forceinline__ void TileMAC(){

    asm volatile(
    "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
        : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
        : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(lhs_fragment[0]), "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));
        
    }
    
    
};

//v2
//仅用于cuda计算， 每个warp负责算两行，结果直接写回global memory
struct cudaComputeUtils_tf32_v2{
    const float * c_rhs_matrix_;
    const int *c_col_indices_;
    const float *c_values_;
    float * output_matrix_;
    int nOri_;
    int mOri_;
    int warpin_id_;
    int warp_id_;
    const int *c_row_offsets_;
    // Constructor
    __device__ __forceinline__ cudaComputeUtils_tf32_v2(
        const float * rhs_matrix,
        const int *c_col_indices,
        const float *c_values,
        float * output_matrix,
        int nOri,
        int mOri,
        int warpin_id,
        int warp_id,
        const int * c_row_offsets
    ):
    c_rhs_matrix_(rhs_matrix),
    c_col_indices_(c_col_indices),
    c_values_(c_values),
    output_matrix_(output_matrix),
    nOri_(nOri),
    mOri_(mOri),
    warpin_id_(warpin_id),
    warp_id_(warp_id),
    c_row_offsets_(c_row_offsets)
    {}
    
    // CUDA Compute
    __device__ __forceinline__ void cudaCompute(int m_index_vec, int dimN_index){
        #pragma unroll
        for(int w=warp_id_*2;w<(warp_id_+1)*2;w++)
        {
            int c_row_offset = m_index_vec*8 + w;
            if(c_row_offset<mOri_){
                int c_row_offset_vec = __ldg(c_row_offsets_ + c_row_offset); 
                int nonzeros = __ldg(c_row_offsets_ + c_row_offset + 1) -  c_row_offset_vec;
                #pragma unroll
                for(int i=0; i<nonzeros; i++)
                {
                    float a = __ldg(c_values_ + c_row_offset_vec + i);
                    int col = __ldg(c_col_indices_ + c_row_offset_vec + i);
                    for(int k=0;k<2;k++)
                    {
                        int c_col_offset = dimN_index + k*32 + warpin_id_;
                        if(c_col_offset<nOri_){
                            float b = __ldg(c_rhs_matrix_ + col*nOri_ + c_col_offset);   
                            *(output_matrix_ + w*nOri_ + dimN_index + k*32 + warpin_id_) +=  a * b;
                        }
                    }
                }
            }
        } 
    }
};