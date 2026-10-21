"""Convert a Hugging Face (transformers) Whisper fine-tune into sherpa-onnx int8 files.
Usage: python3 ci/convert_hf_whisper.py <hf-repo> <out-prefix>
Produces <out-prefix>-encoder.int8.onnx, <out-prefix>-decoder.int8.onnx, <out-prefix>-tokens.txt"""
import importlib.util, os, shutil, sys, urllib.request
import torch, whisper
from transformers import WhisperForConditionalGeneration

repo, prefix = sys.argv[1], sys.argv[2]
hf = WhisperForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.float32)
cfg = hf.config
MAP = [("layers", "blocks"), ("fc1", "mlp.0"), ("fc2", "mlp.2"), ("final_layer_norm", "mlp_ln"),
       (".self_attn.q_proj", ".attn.query"), (".self_attn.k_proj", ".attn.key"), (".self_attn.v_proj", ".attn.value"),
       (".self_attn_layer_norm", ".attn_ln"), (".self_attn.out_proj", ".attn.out"),
       (".encoder_attn.q_proj", ".cross_attn.query"), (".encoder_attn.k_proj", ".cross_attn.key"),
       (".encoder_attn.v_proj", ".cross_attn.value"), (".encoder_attn_layer_norm", ".cross_attn_ln"),
       (".encoder_attn.out_proj", ".cross_attn.out"), ("decoder.layer_norm.", "decoder.ln."),
       ("encoder.layer_norm.", "encoder.ln_post."), ("embed_tokens", "token_embedding"),
       ("encoder.embed_positions.weight", "encoder.positional_embedding"),
       ("decoder.embed_positions.weight", "decoder.positional_embedding")]
sd = {}
for k, v in hf.model.state_dict().items():
    for a, b in MAP:
        k = k.replace(a, b)
    sd[k] = v
dims = dict(n_mels=cfg.num_mel_bins, n_vocab=cfg.vocab_size, n_audio_ctx=cfg.max_source_positions,
            n_audio_state=cfg.d_model, n_audio_head=cfg.encoder_attention_heads, n_audio_layer=cfg.encoder_layers,
            n_text_ctx=cfg.max_target_positions, n_text_state=cfg.d_model, n_text_head=cfg.decoder_attention_heads,
            n_text_layer=cfg.decoder_layers)
print("dims", dims)
m = whisper.model.Whisper(whisper.model.ModelDimensions(**dims))
m.load_state_dict(sd, strict=True)  # fails loudly if the mapping is wrong
torch.save({"dims": dims, "model_state_dict": sd}, "hf-converted.pt")
del hf, m, sd
shutil.rmtree(os.path.expanduser('~/.cache/huggingface'), ignore_errors=True)

url = "https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/master/scripts/whisper/export-onnx.py"
urllib.request.urlretrieve(url, "sherpa_export.py")
spec = importlib.util.spec_from_file_location("sherpa_export", "sherpa_export.py")
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
ex.load_model = lambda name: whisper.load_model("hf-converted.pt")
sys.argv = ["export", "--model", "turbo"]
torch.set_num_threads(4)
from whisper.model import disable_sdpa
with disable_sdpa():
    ex.main()
os.remove("hf-converted.pt")
for part in ["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"]:
    shutil.move(f"turbo-{part}", f"{prefix}-{part}")
    print(prefix + "-" + part, os.path.getsize(f"{prefix}-{part}") // 1_000_000, "MB")
