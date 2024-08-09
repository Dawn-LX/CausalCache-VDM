import os
import gc
import time
from tqdm import tqdm
from datetime import datetime
import torch
import torchvision
import torch.distributed as dist
from colossalai.utils import get_current_device,set_seed
from diffusers.schedulers import LCMScheduler
from opensora.datasets import save_sample
from opensora.registry import SCHEDULERS, build_module
from opensora.utils.misc import to_torch_dtype

from .train_utils import build_progressive_noise

@torch.no_grad()
def validation_visualize(model,vae,text_encoder,val_examples,val_cfgs,exp_dir,writer,global_step):
    gc.collect()
    torch.cuda.empty_cache()
    
    device = get_current_device()
    dtype = to_torch_dtype(val_cfgs.dtype)
    assert vae.patch_size[0] == 1, "TODO: consider temporal patchify"
    
    save_dir = os.path.join(exp_dir,"val_samples",f"{global_step}")
    os.makedirs(save_dir,exist_ok=True)

    scheduler = build_module(val_cfgs.scheduler,SCHEDULERS)
    assert val_cfgs.get("clean_prefix",True), "TODO add code for non first frame conditioned"
    assert val_cfgs.get("clean_prefix_set_t0",True)
    if enable_kv_cache := val_cfgs.get("enable_kv_cache",False):
        sample_func = autoregressive_sample_kv_cache
        additional_kwargs = dict(
            kv_cache_dequeue = val_cfgs.kv_cache_dequeue,
            kv_cache_max_seqlen = val_cfgs.kv_cache_max_seqlen,
        )
        # `kv_cache_max_seqlen` serves as `max_condion_frames` for sampling w/ kv-cache
    else:
        sample_func = autoregressive_sample
        additional_kwargs = dict(
            max_condion_frames = val_cfgs.max_condion_frames
        )
    additional_kwargs.update(dict(
        progressive_alpha=val_cfgs.get("progressive_alpha",-1)
    ))
    
    for idx,example in enumerate(val_examples):
        current_seed = example.seed
        if current_seed == "random":
            current_seed = int(str(datetime.now().timestamp()).split('.')[-1][:4])
        set_seed(current_seed) # TODO 要用generator seet seed 才能每个 example 由自己的seed 唯一确定，否则只是设置了起始seed，与example list的顺序有关

        if (first_image := example.first_image) is not None:
            first_image = first_image.to(device=device,dtype=dtype) # (1,3,h,w)
            cond_frame_latents = vae.encode(first_image.unsqueeze(2)) # vae accept shape (B,C,T,H,W), here B=1,T=1
        else:
            cond_frame_latents = None
        
        # input_size = (example.num_frames, example.height, example.width)
        # latent_size = vae.get_latent_size(input_size)
        assert vae.patch_size[0] == 1
        input_size = (example.auto_regre_chunk_len, example.height, example.width)
        latent_size = vae.get_latent_size(input_size)
    
        samples,time_used,num_gen_frames = sample_func(
            scheduler, 
            model, 
            text_encoder, 
            z_size = (vae.out_channels, *latent_size), 
            prompts = [example.prompt],
            cond_frame_latents=cond_frame_latents, # (B,C,1,H,W)
            ar_steps = example.auto_regre_steps,
            device = device,
            verbose=True,
            **additional_kwargs
        ) # (1, C, T, H, W)
        fps = num_gen_frames / time_used
        print(f"num_gen_frames={num_gen_frames}, time_used={time_used:.2f}, fps={fps:.2f}")
        vae.micro_batch_size = 16
        sample = vae.decode(samples.to(dtype=dtype))[0] # (C, T, H, W)
        vae.micro_batch_size = None

        video_name = f"idx{idx}_seed{current_seed}.mp4"
        save_path = os.path.join(save_dir,video_name)
        save_sample(sample.clone(),fps=8,save_path=save_path)

        if writer is not None:
            low, high = (-1,1)
            sample.clamp_(min=low, max=high)
            sample.sub_(low).div_(max(high - low, 1e-5)) # -1 ~ 1 --> 0 ~ 1
            sample = sample.clamp_(0,1).float().cpu()
            sample = sample.unsqueeze(0).permute(0,2,1,3,4) # BCTHW --> BTCHW

            writer.add_video(
                f"validation-{idx}",
                sample,
                global_step = global_step,
                fps=8,
                walltime=None
            )
    
    gc.collect()
    torch.cuda.empty_cache()

def denormalize(x, value_range=(-1, 1)):

    low, high = value_range
    x.clamp_(min=low, max=high)
    x.sub_(low).div_(max(high - low, 1e-5))
    x = x.mul(255).add_(0.5).clamp_(0, 255)

    return x

# device = next(model.parameters()).device
def autoregressive_sample_kv_cache(
    scheduler, model, text_encoder, 
    z_size, prompts, cond_frame_latents, ar_steps,
    kv_cache_dequeue, kv_cache_max_seqlen, verbose=True,
    **kwargs
):
    
    # cond_frame_latents: (B, C, T_c, H, W)

    bsz = len(prompts)
    c,chunk_len,h,w = z_size
    total_len  = cond_frame_latents.shape[2] + chunk_len * ar_steps
    final_size = (bsz,c,total_len,h,w)

    z_predicted = cond_frame_latents.clone()  # (B,C, T_c, H, W)
    device_dtype = dict(device=z_predicted.device,dtype=z_predicted.dtype)
    do_cls_free_guidance = scheduler.cfg_scale > 1.0
    
    time_start = time.time()
    num_given_frames = z_predicted.shape[2]
    
    model.register_kv_cache(
        bsz*2 if do_cls_free_guidance else bsz,
        max_seq_len = kv_cache_max_seqlen,
        kv_cache_dequeue = kv_cache_dequeue
    )
    if text_encoder is not None:
        model_kwargs = text_encoder.encode(prompts) # {y,mask}
        y_null = text_encoder.null(bsz) if do_cls_free_guidance else None
    else:
        model_kwargs = {"y":None,"mask":None} 
    
    if do_cls_free_guidance:
        model_kwargs["y"] = torch.cat([y_null,model_kwargs["y"]], dim=0)

    model.write_latents_to_cache(
        torch.cat([z_predicted]*2,dim=0) if do_cls_free_guidance else z_predicted,
        **model_kwargs
    )

    init_noise = torch.randn(final_size,**device_dtype)
    progressive_alpha = kwargs.get("progressive_alpha",-1)
    for ar_step in tqdm(range(ar_steps),disable=not verbose):
        predicted_len = z_predicted.shape[2]
        denoise_len = chunk_len
        init_noise_chunk = init_noise[:,:,predicted_len:predicted_len+denoise_len,:,:]
        if progressive_alpha>0: 
            # TODO verify this, check the video gen result is correct
            last_cond = z_predicted[:,:,-1:,:,:]
            tT_bsz = int(scheduler.num_timesteps -1)
            tT_bsz = torch.zeros(size=(bsz,),**device_dtype)
            start_noise = scheduler.q_sample(last_cond,tT_bsz, noise = torch.randn_like(last_cond))
            init_noise_chunk = build_progressive_noise(progressive_alpha, (bsz, *z_size), start_noise)
        
        samples = scheduler.sample_v2(
            model,
            z= init_noise_chunk,
            prompts=prompts,
            device= z_predicted.device,
            model_kwargs = model_kwargs,
            progress_bar = verbose
        ) # (B, C,T_n,H,W)
        
        model.write_latents_to_cache(
            torch.cat([samples]*2,dim=0) if do_cls_free_guidance else samples,
            **model_kwargs
        )

        z_predicted = torch.cat([z_predicted,samples],dim=2) # (B,C, T_accu + T_n, H, W)

        if verbose: 
            print(f"ar_step={ar_step}: given {predicted_len} frames,  denoise:{samples.shape} --> get:{z_predicted.shape}")

    time_used = time.time() - time_start
    num_gen_frames = z_predicted.shape[2] - num_given_frames

    return z_predicted,time_used,num_gen_frames



def autoregressive_sample(
    scheduler, model, text_encoder, 
    z_size, prompts, cond_frame_latents, ar_steps,
    max_condion_frames, verbose=True,
    **kwargs
):
    # cond_frame_latents: (B, C, T_c, H, W)

    bsz = len(prompts)
    c,chunk_len,h,w = z_size
    total_len  = cond_frame_latents.shape[2] + chunk_len * ar_steps
    final_size = (bsz,c,total_len,h,w)

    z_predicted = cond_frame_latents.clone()  # (B,C, T_c, H, W)
    device_dtype = dict(device=z_predicted.device,dtype=z_predicted.dtype)
    do_cls_free_guidance = scheduler.cfg_scale > 1.0
    
    time_start = time.time()
    num_given_frames = z_predicted.shape[2]
    
    if text_encoder is not None:
        model_kwargs = text_encoder.encode(prompts) # {y,mask}
        y_null = text_encoder.null(bsz) if do_cls_free_guidance else None
    else:
        model_kwargs = {"y":None,"mask":None} 
    
    if do_cls_free_guidance:
        model_kwargs["y"] = torch.cat([y_null,model_kwargs["y"]], dim=0)

    
    init_noise = torch.randn(final_size,**device_dtype)
    progressive_alpha = kwargs.get("progressive_alpha",-1)
    for ar_step in tqdm(range(ar_steps),disable=not verbose):
        predicted_len = z_predicted.shape[2]
        denoise_len = chunk_len
        init_noise_chunk = init_noise[:,:,predicted_len:predicted_len+denoise_len,:,:]
        if progressive_alpha > 0: 
            # TODO verify this, check the video gen result is correct
            last_cond = z_predicted[:,:,-1:,:,:]
            tT_bsz = int(scheduler.num_timesteps -1)
            tT_bsz = torch.zeros(size=(bsz,),**device_dtype)
            start_noise = scheduler.q_sample(last_cond,tT_bsz, noise = torch.randn_like(last_cond))
            init_noise_chunk = build_progressive_noise(progressive_alpha, (bsz, *z_size), start_noise)
        
        
        if predicted_len > max_condion_frames:
            print(" >>> condition_frames dequeue")
            z_cond = z_predicted[:,:,-max_condion_frames:,:,:]
        else:
            # predicted_len <=  max_condion_frames, BUT what if predicted_len+denoise_len > max_model_accpet_len ?
            z_cond = z_predicted
        cond_len = z_cond.shape[2]
        z_input = torch.cat([z_cond,init_noise_chunk],dim=2) # (B, C, T_c+T_n, H, W)
        if model.relative_tpe_mode != "cyclic":
            # make sure the temporal position emb not out of range
            assert z_input.shape[2] <= model.temporal_max_len, f'''
            max_condion_frames={max_condion_frames},
            cond_len: {z_cond.shape[2]}, denoise_len: {init_noise_chunk.shape[2]}
            z_input_len = cond_len + denoise_len > model.temporal_max_len = {model.temporal_max_len}
            temporal position embedding (tpe) will out of range !
            '''
            # this happens when (max_condion_frames-first_k_given) % chunk_len !=0, 
            # e.g., max_tpe_len=33, cond: [1,8,9,17,25], chunk_len=8, but we set max_condion_frames=27
        
        if model.relative_tpe_mode is None:
            assert max_condion_frames + denoise_len == model.temporal_max_len 
        else:
            z_input_temporal_start = z_predicted.shape[2] - z_cond.shape[2]
            model_kwargs.update({"x_temporal_start":z_input_temporal_start})

        if not model.is_causal: # TODO ideally remove this, 
            # and find a better way to run baseline's auto-regression in both training & inference
            # 应该直接训练的时候就用不同长度的数据，不要在末尾padding， 比如 9, 17, 25, 33， 每个batch 里面是等长的就好了
            '''NOTE
            For bidirectional attention, the chunk to denoise will be affected by the noise at the end of seq
            e.g., z_cond: [0,1,...,8], denoise_chunk: [9,..,16], noise :[17,...,33]
            each time len(z_input) should exactly equals to model.temporal_max_len

            '''
            num_training_frames = 33
            print(" NOTE: num_training_frames = 33 is hard-coded ", "-="*100)
            if model.relative_tpe_mode is None:
                assert num_training_frames==model.temporal_max_len 
            if z_input.shape[2] < num_training_frames:
                noise_pad_len = num_training_frames - z_input.shape[2]
                print(f" >>> use noise padding: z_len = {z_input.shape[2]} noise_pad_len={noise_pad_len}")
                _b,_c,_t,_h,_w = final_size
                _noise = torch.randn(size=(_b,_c,noise_pad_len,_h,_w),**device_dtype)
                z_input = torch.cat([z_input,_noise],dim=2)
        
        model_kwargs.update({"x_cond":z_cond})
        if model.temp_extra_in_channels > 0: # ideally remove this, the model is aware of clean-prefix using timestep emb
            mask_channel = torch.zeros_like(z_input[:,:1,:,:1,:1]) # (B,1,T,1,1)
            mask_channel[:,:,:cond_len,:,:] = 1
            if do_cls_free_guidance:
                mask_channel = torch.cat([mask_channel]*2, dim=0) #
            model_kwargs.update({"mask_channel":mask_channel})

        samples = scheduler.sample_v2(
            model,
            z= z_input,
            prompts=prompts,
            device= z_predicted.device,
            model_kwargs = model_kwargs,
            progress_bar = verbose
        ) # (B, C,T_c+T_n,H,W)
        if not model.is_causal:
            # samples.shape == (B, C, T, H, W); T is fixed (T== model.temporal_max_len) for each auto-regre step
            pass
        else:
            assert samples.shape[2] == cond_len + denoise_len
        samples = samples[:,:,cond_len:cond_len+denoise_len,:,:]

        z_predicted = torch.cat([z_predicted,samples],dim=2) # (B,C, T_accu + T_n, H, W)

        if verbose: 
            print(f"ar_step={ar_step}: given {predicted_len} frames,  denoise:{samples.shape} --> get:{z_predicted.shape}")

    time_used = time.time() - time_start
    num_gen_frames = z_predicted.shape[2] - num_given_frames

    return z_predicted,time_used,num_gen_frames