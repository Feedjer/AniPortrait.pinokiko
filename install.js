module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: [
        "git clone https://github.com/Zejun-Yang/AniPortrait app",
      ]
    }
  }, {
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        path: "app",
        venv: "env",
        xformers: true
      }
    }
  },{
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
     message: [
     "pip install -r requirements.txt"
      ]
    }
 },
 // {
  //  method: "fs.link",
   // params: {
    //  venv: "app/env"
   //   }
  // },
  {
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/ZJYang/AniPortrait/resolve/main/audio2mesh.pt",
      dir: "app/pretrained_model"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/ZJYang/AniPortrait/resolve/main/audio2pose.pt",
      dir: "app/pretrained_model"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/ZJYang/AniPortrait/resolve/main/denoising_unet.pth",
      dir: "app/pretrained_model"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/ZJYang/AniPortrait/resolve/main/film_net_fp16.pt",
      dir: "app/pretrained_model"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/ZJYang/AniPortrait/resolve/main/motion_module.pth",
      dir: "app/pretrained_model"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/ZJYang/AniPortrait/resolve/main/pose_guider.pth",
      dir: "app/pretrained_model"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/ZJYang/AniPortrait/resolve/main/reference_unet.pth",
      dir: "app/pretrained_model"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/feature_extractor/preprocessor_config.json?download=true",
      dir: "app/pretrained_model/stable-diffusion-v1-5/feature_extractor"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/model_index.json?download=true",
      dir: "app/pretrained_model/stable-diffusion-v1-5"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/config.json?download=true",
      dir: "app/pretrained_model/stable-diffusion-v1-5/unet"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin?download=true",
      dir: "app/pretrained_model/stable-diffusion-v1-5/unet"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-inference.yaml?download=true",
      dir: "app/pretrained_model/stable-diffusion-v1-5"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json?download=true",
      dir: "app/pretrained_model/sd-vae-ft-mse"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin?download=true",
      dir: "app/pretrained_model/sd-vae-ft-mse"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors?download=true",
      dir: "app/pretrained_model/sd-vae-ft-mse"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/config.json?download=true",
      dir: "app/pretrained_model/image_encoder"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/pytorch_model.bin",
      dir: "app/pretrained_model/image_encoder"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json?download=true",
      dir: "app/pretrained_model/wav2vec2-base-960h"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/feature_extractor_config.json?download=true",
      dir: "app/pretrained_model/wav2vec2-base-960h"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/preprocessor_config.json?download=true",
      dir: "app/pretrained_model/wav2vec2-base-960h"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin?download=true",
      dir: "app/pretrained_model/wav2vec2-base-960h"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/README.md?download=true",
      dir: "app/pretrained_model/wav2vec2-base-960h"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/special_tokens_map.json?download=true",
      dir: "app/pretrained_model/wav2vec2-base-960h"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/tokenizer_config.json?download=true",
      dir: "app/pretrained_model/wav2vec2-base-960h"
    }
  },{
    method: "fs.download",
    params: {
      uri: "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json?download=true",
      dir: "app/pretrained_model/wav2vec2-base-960h"
    }
  },{
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
     message: [
     "pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121"
      ]
    }
 },{
  method: "shell.run",
  params: {
    path: "app/pretrained_model/stable-diffusion-v1-5/feature_extractor",
   message: [
   "ren feature_extractor%2Fpreprocessor_config.json config.json"
    ]
  }
},{
  method: "shell.run",
  params: {
    path: "app/pretrained_model/image_encoder",
   message: [
   "ren image_encoder%2Fconfig.json config.json"
    ]
  }
},{
  method: "shell.run",
  params: {
    path: "app/pretrained_model/stable-diffusion-v1-5/unet",
   message: [
   "ren unet%2Fconfig.json config.json"
    ]
  }
},{
    method: "notify",
    params: {
      html: "Click the 'start' tab to get started!"
    }
  }]
}
