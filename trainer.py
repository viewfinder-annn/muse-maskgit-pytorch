import os
import json
import time
import random
import json5
import torch
import torchaudio
import dac
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DistributedType
from accelerate import DistributedDataParallelKwargs
from transformers import get_scheduler as get_transformers_scheduler
from muse_maskgit_pytorch import MaskGit, MaskGitTransformer, AudioEncoder
from dataset.degradation_dataset import AudioDegradationDataset
from torch.nn.parallel import DistributedDataParallel as DDP

# for w2v-bert 2.0
from transformers import AutoFeatureExtractor, AutoModel

def pad_or_truncate(x, length=512*256):
    if x.size(-1) < length:
        x = torch.nn.functional.pad(x, (0, length - x.size(-1)))
    elif x.size(-1) > length:
        x = x[..., :length]
    return x

def get_dataloader(config, device):
    dataset = AudioDegradationDataset(speech_list=config['train']['speech_list'], noise_list=config['train']['noise_list'], rir_list=config['train']['rir_list'], degradation_config=config['degradation_config'], seq_len=config.get('seq_len', 512*256), sr=config['sample_rate'], device=device)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    # TODO: val dataset
    return dataloader

def get_model(config, device):
    # Load DAC model
    dac_model = dac.DAC.load(config['dac_path']).to(device)
    dac_model.to(device)
    dac_model.eval()
    dac_model.requires_grad_(False)

    # Initialize transformer and audio encoder
    transformer_config = config['MaskGitTransformer']
    audio_encoder_config = config['AudioEncoder']
    transformer = MaskGitTransformer(**transformer_config)
    audio_encoder = AudioEncoder(**audio_encoder_config)

    # Initialize MaskGit model
    maskgit_config = config['MaskGit']
    model = MaskGit(
        vq_model=dac_model,
        transformer=transformer,
        audio_encoder=audio_encoder,
        **maskgit_config
    ).to(device)
    
    print(f"model Params: {round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 2)}M")

    return model

def get_optimizer(model, config):
    if config['optimizer'] == 'adam':     
        # Retrieve specific parameters for Adam optimizer
        optimizer_config = config['adam']
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            betas=tuple(optimizer_config['betas']),
            eps=optimizer_config['eps']
        )
        return optimizer
    else:
        raise NotImplementedError(f"Optimizer {config['optimizer']} not implemented")

def get_scheduler(optimizer, config):
    if 'scheduler' not in config:
        return None
    elif config['scheduler'] == 'linear':
        scheduler_config = config['linear']
        print(f"current device count: {torch.cuda.device_count()}")
        scheduler = get_transformers_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=scheduler_config['num_warmup_steps'] * torch.cuda.device_count(),
            num_training_steps=scheduler_config['num_training_steps'] * torch.cuda.device_count()
        )
        return scheduler
    else:
        raise NotImplementedError(f"Scheduler {config['scheduler']} not implemented")

def train_loop(config, exp_name, model, dataloader, optimizer, scheduler, device, epochs=10, test_noisy_path=None, save_every_step=1000, eval_every_step=1000, resume_path=None):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    print(f"train_loop {resume_path}")
    
    model_dst = f'./exp/{exp_name}/model'
    os.makedirs(model_dst, exist_ok=True)
    audio_dst = f'./exp/{exp_name}/output'
    os.makedirs(audio_dst, exist_ok=True)
    
    input_noisy_paths = random.sample([os.path.join(test_noisy_path, f) for f in os.listdir(test_noisy_path) if f.endswith('.wav')], 10)
    if accelerator.is_main_process:
        writer = SummaryWriter(f'./exp/{exp_name}/logs')
    else:
        writer = None
    
    if epochs < 0:
        epochs = int(1e9)

    if resume_path is not None:
        print(f"Resuming training from {resume_path}")
        checkpoint = torch.load(os.path.join(resume_path, 'checkpoint.pth'), map_location=device)
        model.load_state_dict(torch.load(os.path.join(resume_path, checkpoint['model']), map_location=device))
        optimizer.load_state_dict(torch.load(os.path.join(resume_path, checkpoint['optimizer']), map_location=device))
        if checkpoint['scheduler'] is not None:
            scheduler.load_state_dict(torch.load(os.path.join(resume_path, checkpoint['scheduler']), map_location=device))
        global_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
    else:
        global_epoch = 0
        global_step = 0
    
    if accelerator.unwrap_model(model).return_audio_embed:
        semantic_feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        semantic_model = AutoModel.from_pretrained("facebook/w2v-bert-2.0")
        semantic_model.to(device)
        semantic_model.eval()
        resampler = torchaudio.transforms.Resample(44100, 16000).to(device)
        
        def resolution_transformation(content, target_len, target_dim=None):
            """
            Transform the resolution of the input content to match the target length.
            
            Args:
                content: torch.tensor, shape (batch_size, source_len, dim)
                target_len: int, target length
            
            Returns:
                mapped_feature: torch.tensor, shape (target_len, dim)
            """

            batch_size, source_len, width = content.shape
            if target_dim is not None:
                width = target_dim
            
            content_4d = content.unsqueeze(1)
            
            # 使用插值进行上采样或下采样，对每个batch进行处理
            mapped_feature_tensor = torch.nn.functional.interpolate(content_4d, size=(target_len, width), mode='bilinear', align_corners=False)
            
            mapped_feature_tensor = mapped_feature_tensor.squeeze(1)
            
            return mapped_feature_tensor
    
    save_part_loss = 0
    eval_part_loss = 0

    for epoch in range(global_epoch, epochs):
        sum_loss = 0
        model.train()
        if accelerator.is_main_process:
            dataloader = tqdm(dataloader)
        for clean_signals, noisy_signals in dataloader:
            clean_signals = clean_signals.to(device) # [batch_size, 1, seq_len]
            noisy_signals = noisy_signals.to(device)
            if not accelerator.unwrap_model(model).return_audio_embed:
                loss = model(clean_audios=clean_signals, noisy_audios=noisy_signals)  # Update according to actual model signature
                
                if writer is not None:
                    writer.add_scalar('Step/cross_entropy_loss', loss.item(), global_step)
                
                # loss.backward()
                accelerator.backward(loss)
            else:
                loss, audio_embed = model(clean_audios=clean_signals, noisy_audios=noisy_signals) # audio_embed: [b, n, dim]
                clean_audios_16k = resampler(clean_signals)
                clean_audios_16k = clean_audios_16k.cpu() # [batch_size, 1, seq_len_16k]
                input_features = torch.cat([
                    semantic_feature_extractor(audio, sampling_rate=16000, return_tensors='pt', padding=True)['input_features']
                    for audio in clean_audios_16k
                ], dim=0) # [batch_size, 136, 160] (80 band mel * 2)
                with torch.no_grad():
                    outputs = semantic_model(input_features=input_features.to(device), output_hidden_states=True)
                    semantic_features = outputs.hidden_states[17] # [batch_size, 136, 1024]
                semantic_features = resolution_transformation(semantic_features, audio_embed.shape[1], audio_embed.shape[2])
                
                # L2 loss
                encoder_loss = torch.nn.functional.mse_loss(semantic_features, audio_embed)
                # encoder_loss_ratio = 0.1
                encoder_loss_ratio = 1
                
                if writer is not None:
                    writer.add_scalar('Step/encoder_loss', encoder_loss.item(), global_step)
                    writer.add_scalar('Step/cross_entropy_loss', loss.item(), global_step)
                
                loss = loss + encoder_loss_ratio * encoder_loss
                # loss.backward()
                accelerator.backward(loss)
                
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            
            if writer is not None:
                # add step lr & loss to tensorboard
                writer.add_scalar('Step/loss', loss.item(), global_step)
                writer.add_scalar('Step/lr', optimizer.param_groups[0]['lr'], global_step)

            sum_loss += loss.item()
            save_part_loss += loss.item()
            eval_part_loss += loss.item()
            global_step += 1

            # if global_step % 100 == 0:  # 每100步检查一次
            #     current_lr = optimizer.param_groups[0]['lr']
            #     print(f"Process {accelerator.process_index}, Step {global_step}, LR: {current_lr}")
            
            if global_step % eval_every_step == 0 and accelerator.is_main_process:
                if input_noisy_paths is not None:
                    with torch.no_grad():
                        noisy_audios = []
                        for input_path in input_noisy_paths:
                            signal, sr = torchaudio.load(input_path)
                            signal = signal.to(device)
                            # resample to 44.1k
                            signal = torchaudio.transforms.Resample(sr, config["dataset"]["sample_rate"]).to(device)(signal)
                            signal = pad_or_truncate(signal, length=config["dataset"]["seq_len"])
                            noisy_audios.append(signal)
                        noisy_audios = torch.stack(noisy_audios)
                        noisy_audios.squeeze_(1)
                        noisy_audios = noisy_audios.to(device)
                        output_dir = f'{audio_dst}/epoch-{epoch}-step-{global_step}-loss-{round(eval_part_loss/eval_every_step, 4)}'
                        os.makedirs(output_dir, exist_ok=True)
                        # 分批处理
                        for i in range(noisy_audios.shape[0]):
                            noisy_audio = noisy_audios[i].unsqueeze(0)
                            ids, clean_audios = accelerator.unwrap_model(model).generate(noisy_audio)
                            torchaudio.save(f'{output_dir}/noisy_{i}.wav', noisy_audio.detach().cpu(), config["dataset"]["sample_rate"])
                            torchaudio.save(f'{output_dir}/noisy_{i}_enhanced.wav', clean_audios[0].detach().cpu(), config["dataset"]["sample_rate"])
                eval_part_loss = 0

            accelerator.wait_for_everyone()
            if global_step % save_every_step == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch} Step {global_step} Loss: {save_part_loss/save_every_step}")
                model_name = f"epoch-{epoch}-step-{global_step}-loss-{round(save_part_loss/save_every_step, 4)}"
                save_path = os.path.join(model_dst, model_name)
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
                torch.save(optimizer.state_dict(), os.path.join(save_path, 'optimizer.pt'))
                if scheduler is not None:
                    torch.save(scheduler.state_dict(), os.path.join(save_path, 'scheduler.pt'))
                checkpoint = {
                    'model': 'model.pt',
                    'optimizer': 'optimizer.pt',
                    'scheduler': 'scheduler.pt' if scheduler is not None else None,
                    'epoch': epoch,
                    'global_step': global_step
                }
                torch.save(checkpoint, os.path.join(save_path, 'checkpoint.pth'))
                save_part_loss = 0

        if writer is not None:
            # add epoch loss to tensorboard
            writer.add_scalar('Epoch/loss', sum_loss/len(dataloader), epoch)
        print(f"Epoch {epoch} Finished, Avg Loss: {sum_loss/len(dataloader)}")

    if writer is not None:
        writer.close()
    return model

def main(config_path, resume_path=None):
    print(f"main {resume_path}")
    # Load configuration
    with open(config_path, 'r') as f:
        config = json5.load(f)
    exp_name = f"{time.strftime('%Y%m%d-%H:%M')}-{os.path.basename(config_path).replace('.json', '')}"
    exp_dst = f'./exp/{exp_name}'
    os.makedirs(exp_dst, exist_ok=True)
    with open(f'{exp_dst}/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    device = config['train']['device']
    dataloader = get_dataloader(config['dataset'], device)
    model = get_model(config['model'], device)
    optimizer = get_optimizer(model, config['train'])
    scheduler = get_scheduler(optimizer, config['train'])
    model = train_loop(config, exp_name, model, dataloader, optimizer, scheduler, device, config['train']['epochs'], config['dataset']['test_noisy_path'], config['train']['save_every_step'], config['train']['eval_every_step'], resume_path)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model based on the given configuration file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to the model to resume training from.')
    args = parser.parse_args()
    print(f"args {args}")
    main(args.config, args.resume_path)