# AudioCraft
AudioCraft is a PyTorch library for deep learning research on audio generation. AudioCraft contains inference and training code for two state-of-the-art AI generative models producing high-quality audio: AudioGen and MusicGen.     

# Models
At the moment, AudioCraft contains the training code and inference code for:

- `**MusicGen**`: A state-of-the-art controllable text-to-music model.
- `**AudioGen**`: A state-of-the-art text-to-sound model.
- `**EnCodec**`: A state-of-the-art high fidelity neural audio codec.
- `**Multi Band Diffusion**`: An EnCodec compatible decoder using diffusion.

# Training code
- This is a very simple notebook in which I use audiocraft model from Facebook Research to generate music
- Just playing with the model and Generative AI for audio
- [Audio Craft paper - Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284)
- Using [model](https://huggingface.co/facebook/musicgen-small) from 🤗 Hugging Face

# References
1. [API Documentaton](https://facebookresearch.github.io/audiocraft/api_docs/audiocraft/index.html)
2. [Blog](https://about.fb.com/news/2023/08/audiocraft-generative-ai-for-music-and-audio/)
