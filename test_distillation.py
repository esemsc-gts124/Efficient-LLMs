#!/usr/bin/env python3
"""
Simple test script to verify inter-process group communication.
Run with: torchrun --nproc_per_node=3 test_distillation.py
(2 training ranks + 1 distillation rank)
"""

import torch
import torch.distributed as dist
from lingua.distributed import (
    DistributedArgs,
    InterProcessComm,
    setup_torch_distributed,
    get_global_rank,
    get_world_size,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_communication():
    # Setup distributed
    dist_args = DistributedArgs()
    setup_torch_distributed(dist_args)
    
    # Create IPC helper
    ipc = InterProcessComm(dist_args)
    
    logger.info(f"Rank {get_global_rank()}/{get_world_size()} - is_training: {ipc.is_training}")
    
    if ipc.is_training:
        # Training process
        logger.info(f"[TEST] Training rank {get_global_rank()} waiting for model sync...")
        ipc.sync_model_loaded()
        logger.info(f"[TEST] Training rank {get_global_rank()} model sync complete!")
        
        # Create dummy batch
        batch_tokens = torch.randint(0, 1000, (4, 128))  # batch_size=4, seq_len=128
        logger.info(f"[TEST] Training rank {get_global_rank()} sending batch shape {batch_tokens.shape}")
        
        # Send shape coordination
        if get_global_rank() == 0:
            ipc.send_batch_shape([4, 128])
        else:
            ipc.send_batch_shape(None)
        
        # Send batch
        ipc.gather_batch_to_distill(batch_tokens)
        
        # Receive output
        output = ipc.scatter_tensors_from_distill()
        logger.info(f"[TEST] Training rank {get_global_rank()} received output shape: {output.shape if output is not None else None}")
        
    else:
        # Distillation process
        import time
        logger.info(f"[TEST] Distill rank simulating model load...")
        time.sleep(1)
        
        # Signal model ready
        ipc.sync_model_loaded()
        logger.info(f"[TEST] Distill rank signaled model ready")
        
        # Receive batches
        gathered = ipc.gather_batch_to_distill(None)
        logger.info(f"[TEST] Distill rank received {len(gathered)} batches")
        
        # Create dummy outputs
        outputs = []
        for batch in gathered:
            output = torch.randn(batch.shape[0], 256, device='cuda')
            outputs.append(output)
        
        # Send back outputs
        ipc.scatter_tensors_from_distill(outputs)
        logger.info(f"[TEST] Distill rank sent outputs back")
    
    logger.info(f"[TEST] Rank {get_global_rank()} test complete!")


if __name__ == "__main__":
    test_communication()