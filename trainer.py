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
from transformers import get_scheduler as get_transformers_scheduler
from muse_maskgit_pytorch import MaskGit, MaskGitTransformer, AudioEncoder
from dataset.degradation_dataset import AudioDegradationDataset

def pad_or_truncate(x, length=512*256):
    if x.size(-1) < length:
        x = torch.nn.functional.pad(x, (0, length - x.size(-1)))
    elif x.size(-1) > length:
        x = x[..., :length]
    return x

class AudioDataset(Dataset):
    def __init__(self, data_path, sr=44100, device='cuda'):
        assert os.path.exists(data_path), f'{data_path} does not exist'
        assert os.path.exists(os.path.join(data_path, 'clean')), f'{data_path}/clean does not exist'
        self.clean_path = os.path.join(data_path, 'clean')
        self.clean_paths = [os.path.join(self.clean_path, f) for f in os.listdir(self.clean_path) if f.endswith('.wav')]
        self.noisy_paths = [i.replace('clean', 'noisy') for i in self.clean_paths]
        self.sr = sr
        self.device = device

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_signal, _ = torchaudio.load(self.clean_paths[idx])
        noisy_signal, _ = torchaudio.load(self.noisy_paths[idx])
        clean_signal = clean_signal.to(self.device)
        noisy_signal = noisy_signal.to(self.device)

        # Resample and pad/truncate both signals
        clean_signal = torchaudio.transforms.Resample(orig_freq=_, new_freq=self.sr).to(self.device)(clean_signal)
        noisy_signal = torchaudio.transforms.Resample(orig_freq=_, new_freq=self.sr).to(self.device)(noisy_signal)

        clean_signal = pad_or_truncate(clean_signal, 512*256)
        noisy_signal = pad_or_truncate(noisy_signal, 512*256)

        return clean_signal, noisy_signal

def get_dataloader(config, device):
    dataset = AudioDegradationDataset(speech_list=config['train']['speech_list'], noise_list=config['train']['noise_list'], rir_list=config['train']['rir_list'], degradation_config=config['degradation_config'], seq_len=512*256, sr=44100, device=device)
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
        scheduler = get_transformers_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=scheduler_config['num_warmup_steps'],
            num_training_steps=scheduler_config['num_training_steps']
        )
        return scheduler
    else:
        raise NotImplementedError(f"Scheduler {config['scheduler']} not implemented")

def train_loop(exp_name, model, dataloader, optimizer, scheduler, device, epochs=10, test_noisy_path=None, save_every_step=1000, eval_every_step=1000, resume_path=None):
    # accelerator = Accelerator()
    # model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    print(f"train_loop {resume_path}")
    
    model_dst = f'./exp/{exp_name}/model'
    os.makedirs(model_dst, exist_ok=True)
    audio_dst = f'./exp/{exp_name}/output'
    os.makedirs(audio_dst, exist_ok=True)
    
    input_noisy_paths = random.sample([os.path.join(test_noisy_path, f) for f in os.listdir(test_noisy_path) if f.endswith('.wav')], 20)
    writer = SummaryWriter(f'./exp/{exp_name}/logs')
    
    if epochs < 0:
        epochs = int(1e9)

    if resume_path is not None:
        print(f"Resuming training from {resume_path}")
        resume_model_name = resume_path.split('/')[-1]
        # epoch-0-step-1000-loss-6.531188071727753
        epoch = int(resume_model_name.split('-')[1])
        global_step = int(resume_model_name.split('-')[3])
        epochs = epochs - epoch
        model.load_state_dict(torch.load(os.path.join(resume_path, 'model.pt')))
        optimizer.load_state_dict(torch.load(os.path.join(resume_path, 'optimizer.pt')))
        if os.path.exists(os.path.join(resume_path, 'scheduler.pt')):
            scheduler.load_state_dict(torch.load(os.path.join(resume_path, 'scheduler.pt')))
    else:
        global_step = 0
    
    save_part_loss = 0
    eval_part_loss = 0

    for epoch in range(epochs):
        sum_loss = 0
        model.train()
        for batch_idx, (clean_signals, noisy_signals) in tqdm(enumerate(dataloader), total=len(dataloader)):
            clean_signals = clean_signals.to(device)
            noisy_signals = noisy_signals.to(device)
            loss = model(clean_audios=clean_signals, noisy_audios=noisy_signals)  # Update according to actual model signature
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            
            # add step lr & loss to tensorboard
            writer.add_scalar('Step/loss', loss.item(), global_step)
            writer.add_scalar('Step/lr', optimizer.param_groups[0]['lr'], global_step)

            sum_loss += loss.item()
            save_part_loss += loss.item()
            eval_part_loss += loss.item()
            global_step += 1

            if global_step % eval_every_step == 0:
                if input_noisy_paths is not None:
                    with torch.no_grad():
                        noisy_audios = []
                        for input_path in input_noisy_paths:
                            signal, sr = torchaudio.load(input_path)
                            signal = signal.to(device)
                            # resample to 44.1k
                            signal = torchaudio.transforms.Resample(sr, 44100).to(device)(signal)
                            signal = pad_or_truncate(signal)
                            noisy_audios.append(signal)
                        noisy_audios = torch.stack(noisy_audios)
                        noisy_audios.squeeze_(1)
                        noisy_audios = noisy_audios.to(device)
                        ids, clean_audios = model.generate(noisy_audios, timesteps=40)
                        output_dir = f'{audio_dst}/epoch-{epoch}-step-{global_step}-loss-{eval_part_loss/eval_every_step}'
                        os.makedirs(output_dir, exist_ok=True)
                        for i in range(10):
                            torchaudio.save(f'{output_dir}/noisy_{i}_enhanced.wav', clean_audios[i].detach().cpu(), 44100)
                            torchaudio.save(f'{output_dir}/noisy_{i}.wav', noisy_audios[i].unsqueeze(0).detach().cpu(), 44100)
                eval_part_loss = 0

            if global_step % save_every_step == 0:
                print(f"Epoch {epoch} Step {global_step} Loss: {save_part_loss/save_every_step}")
                model_name = f"epoch-{epoch}-step-{global_step}-loss-{save_part_loss/save_every_step}"
                save_path = os.path.join(model_dst, model_name)
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
                torch.save(optimizer.state_dict(), os.path.join(save_path, 'optimizer.pt'))
                torch.save(scheduler.state_dict(), os.path.join(save_path, 'scheduler.pt'))
                save_part_loss = 0

        # add epoch loss to tensorboard
        writer.add_scalar('Epoch/loss', sum_loss/len(dataloader), epoch)
        print(f"Epoch {epoch} Finished, Avg Loss: {sum_loss/len(dataloader)}")

    writer.close()
    return model

def main(config_path, resume_path=None):
    print(f"main {resume_path}")
    # Load configuration
    with open(config_path, 'r') as f:
        config = json5.load(f)
    exp_name = f"{os.path.basename(config_path).replace('.json', '')}-{time.strftime('%Y%m%d-%H:%M')}"
    exp_dst = f'./exp/{exp_name}'
    os.makedirs(exp_dst, exist_ok=True)
    with open(f'{exp_dst}/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    device = config['train']['device']
    dataloader = get_dataloader(config['dataset'], device)
    model = get_model(config['model'], device)
    optimizer = get_optimizer(model, config['train'])
    scheduler = get_scheduler(optimizer, config['train'])
    model = train_loop(exp_name, model, dataloader, optimizer, scheduler, device, config['train']['epochs'], config['dataset']['test_noisy_path'], config['train']['save_every_step'], config['train']['eval_every_step'], resume_path)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model based on the given configuration file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to the model to resume training from.')
    args = parser.parse_args()
    print(f"args {args}")
    main(args.config, args.resume_path)