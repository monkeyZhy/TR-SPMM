
#include <mma.h>
#include <cstdint>
#include <stdio.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
    //Tile_N = 128 threads_per_block = 128
    struct mmaDenseTile_fp16_map{
        const float *  values_;
        const int2 *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const half2 *matrix_base_;
        half *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_fp16_map(
        long row_offset_vec,
        const float * values,
        const int2 *  column_idxs,
	    int rhs_cols,
        int offset, 
        int lane_id, 
        const half2*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //每行16个线程，每个线程搬运1个double,
            matrix_base_(matrix + offset),
            // row_offsets_base_(row_offsets),
            values_(values + row_offset_vec*4 + (lane_id & 31)),
            column_idxs_(column_idxs + row_offset_vec/2 + (((lane_id & 31)%4))),
            dense_tile_(reinterpret_cast< half *>(dense_tile)),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]=__ldg(values_);
            values_ += 32;
            int2 col_temp = __ldg(column_idxs_);
            column_idxs_ += 4;

            //load dense
            const int global_offset = (warp_id<<3) + (warpin_id/4);
            long offset = (col_temp.x*rhs_cols_/2)+ global_offset;
            half2 temp = __ldg(matrix_base_ +offset);
            dense_tile_[0]= temp.x;
            dense_tile_[2]= temp.y;

            offset = (col_temp.y*rhs_cols_/2)+ global_offset;
            temp = __ldg(matrix_base_ +offset);
            dense_tile_[1]= temp.x;
            dense_tile_[3]= temp.y;
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index, int residue){

            sparse_fragment_[0]=__ldg(values_);
            int2 col_temp = __ldg(column_idxs_);

            //load dense
            const int global_offset = (warp_id<<3) + (warpin_id/4);
            if(col_temp.x>=0)
            {            
                long offset = (col_temp.x*rhs_cols_/2)+ global_offset;
                half2 temp = __ldg(matrix_base_ +offset);
                dense_tile_[0]= temp.x;
                dense_tile_[2]= temp.y;
            }
            if(col_temp.y>=0)
            {            
                long offset = (col_temp.y*rhs_cols_/2)+ global_offset;
                half2 temp = __ldg(matrix_base_ +offset);
                dense_tile_[1]= temp.x;
                dense_tile_[3]= temp.y;
            }
        }
    };


    //Tile_N = 128 threads_per_block = 128
    struct mmaDenseTile_tf32{
        const float *  values_;
        const int *  column_idxs_;
        const int rhs_cols_;
        const int lane_id_;
        const int warpin_id;
        const int warp_id;
        const float2 *matrix_base_;
        float2 *dense_tile_;
        //存放当前线程拿到的一个double值，并将该double值拆分成4个half分别进行转置放置
        float *sparse_fragment_;

        __device__ __forceinline__ mmaDenseTile_tf32(
        long row_offset_vec,
        const float * values,
        const int *  column_idxs,
	    int rhs_cols,
        int offset1, 
        int lane_id, 
        const float*  matrix, 
        //row_offsets= column_indices_tile
        // const int *row_offsets,
        float2 * dense_tile,
        float *sparse_fragment):
            rhs_cols_(rhs_cols),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5),
            //当前block在全局的列偏移
            matrix_base_(reinterpret_cast<const float2 *>(matrix + offset1)),
            //8的意思是vector的长度
            values_((values + row_offset_vec*8) + (lane_id & 31)),
            //对4*16的RHS读取，每行连续读8个线程，共4行，所以需要>>3
            column_idxs_(column_idxs + row_offset_vec + ((lane_id & 31)%4)),
            dense_tile_(dense_tile),
            sparse_fragment_(sparse_fragment)
            {}
    
        __device__ __forceinline__ void Fetch(int colEdge, int dimN_index){

            sparse_fragment_[0]= __ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);
            values_ += 32;
            column_idxs_ += 4;
            // (warp_id<<4) 每个warp有16列
            //行偏移,(warpin_id%8)*2),每行8个线程，每个线程读两个float数
            const int global_offset = (warp_id<<3) + ((warpin_id%8));
            const long offset = (row_offsets_*rhs_cols_/2) + global_offset;
            dense_tile_[0] =__ldg(matrix_base_ +offset);
        }

        // Load the residual and compute the matrix product
        __device__ __forceinline__ void ResidueLoad(int colEdge, int dimN_index){
            sparse_fragment_[0]= __ldg(values_);
            const long row_offsets_ = __ldg(column_idxs_);

            if(row_offsets_ >= 0)
            { // (warp_id<<4) 每个warp有16列
            //行偏移,(warpin_id%8)*2),每行8个线程，每个线程读两个float数
            const int global_offset = (warp_id<<3) + ((warpin_id%8));
            const long offset = (row_offsets_*rhs_cols_/2) + global_offset;
            dense_tile_[0] =__ldg(matrix_base_ +offset);
            }
        }
    };