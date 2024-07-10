import torchaudio
import torch
from muse_maskgit_pytorch import MaskGit, MaskGitTransformer, AudioEncoder
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import json5
import json
import time
import dac
import random
from transformers import get_scheduler as get_transformers_scheduler
from torch.utils.tensorboard import SummaryWriter

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
    dataset = AudioDataset(data_path=config['train_data_path'], sr=config['sample_rate'], device=device)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
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

def train_loop(exp_name, model, dataloader, optimizer, scheduler, device, epochs=10, test_noisy_path=None, save_every=5):
    model_dst = f'./exp/{exp_name}/model'
    os.makedirs(model_dst, exist_ok=True)
    audio_dst = f'./exp/{exp_name}/output'
    os.makedirs(audio_dst, exist_ok=True)
    
    input_noisy_paths = random.sample([os.path.join(test_noisy_path, f) for f in os.listdir(test_noisy_path) if f.endswith('.wav')], 10)
    
    writer = SummaryWriter(f'./exp/{exp_name}/logs')
    
    for epoch in range(epochs):
        sum_loss = 0
        part_loss = 0
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
            writer.add_scalar('Step/loss', loss.item(), epoch*len(dataloader)+batch_idx)
            writer.add_scalar('Step/lr', optimizer.param_groups[0]['lr'], epoch*len(dataloader)+batch_idx)
            
            sum_loss += loss.item()
            part_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Part Loss: {part_loss/10}, Avg Loss: {sum_loss/(batch_idx+1)}")
                part_loss = 0
        
        # add epoch loss to tensorboard
        writer.add_scalar('Epoch/loss', sum_loss/len(dataloader), epoch)
        print(f"Epoch {epoch} Finished, Avg Loss: {sum_loss/len(dataloader)}")
        
        if epoch % save_every == 0:
            model_name = f"epoch-{epoch}-loss-{sum_loss/len(dataloader)}.pt"
            model.save(f'{model_dst}/{model_name}')
        
        if input_noisy_paths is not None:
            with torch.no_grad():
                noisy_audios = []
                for input_path in input_noisy_paths:
                    signal, sr = torchaudio.load(input_path)
                    signal = signal.to(device)
                    # resample to 44.1k
                    signal = torchaudio.transforms.Resample(sr, 44100).to(device)(signal)
                    # print(signal.shape, signal.device)
                    signal = pad_or_truncate(signal)
                    noisy_audios.append(signal)
                noisy_audios = torch.stack(noisy_audios)
                noisy_audios.squeeze_(1)
                noisy_audios = noisy_audios.to(device)
                ids, clean_audios = model.generate(noisy_audios, timesteps=40)
                output_dir = f'{audio_dst}/epoch-{epoch}-loss-{sum_loss/len(dataloader)}'
                os.makedirs(output_dir, exist_ok=True)
                for i in range(10):
                    print(clean_audios[i].shape, noisy_audios[i].shape)
                    torchaudio.save(f'{output_dir}/noisy_{i}_enhanced.wav', clean_audios[i].detach().cpu(), 44100)
                    torchaudio.save(f'{output_dir}/noisy_{i}.wav', noisy_audios[i].unsqueeze(0).detach().cpu(), 44100)
    
    writer.close()  # 关闭Writer
    return model

def main(config_path):
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
    model = train_loop(exp_name, model, dataloader, optimizer, scheduler, device, config['train']['epochs'], config['dataset']['test_noisy_path'], config['train']['save_every'])
    return model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a model based on the given configuration file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    main(args.config)