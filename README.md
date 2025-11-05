# Multi-Objective Balanced Covering for Visual Token Pruning (MoB)

<p align="center">
  <img src="images/teaser.png" alt="Overview of the Multi-Objective Balanced covering framework" width="90%" />
</p>

Official implementation of the NeurIPS 2025 paper **"Why 1 + 1 < 1 in Visual Token Pruning: Beyond Naïve Integration via Multi-Objective Balanced Covering"**. MoB introduces an adaptive, geometry-aware pruning strategy that jointly optimizes **prompt alignment** and **visual coverage** so that multimodal large language models (MLLMs) retain only the most informative image tokens without sacrificing accuracy.

---

## Table of Contents
- [Highlights](#highlights)
- [MoB at a Glance](#mob-at-a-glance)
- [Quantitative & Qualitative Results](#quantitative--qualitative-results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Environment Setup](#environment-setup)
  - [Using MoB with LLaVA](#using-mob-with-llava)
  - [Using MoB with Qwen2-VL](#using-mob-with-qwen2-vl)
  - [Benchmarking with lmms-eval](#benchmarking-with-lmms-eval)
- [Model Zoo](#model-zoo)
- [Citation](#citation)
- [Acknowledgements & License](#acknowledgements--license)

---

## Highlights
- **Multi-objective balanced covering.** MoB performs prompt-aware clustering followed by farthest point sampling to balance semantic alignment and spatial coverage when pruning visual tokens inside transformer decoders.
- **Drop-in modules for popular MLLMs.** We provide MoB-enabled decoders for both **LLaVA** and **Qwen2-VL**, including configuration hooks that expose pruning hyper-parameters at inference time.
- **End-to-end evaluation support.** Modified `lmms-eval` pipelines capture per-sample retained patch indices so you can audit pruning decisions alongside standard benchmark metrics.
- **General-purpose toolkit.** The repository packages ready-to-use scripts, configs, and figures for reproducing the paper’s results across efficiency–accuracy trade-offs.

---

## MoB at a Glance

### Balanced covering strategy
MoB attaches to the vision-text decoder and triggers once per chosen layer. Given hidden states at that layer, the algorithm:
1. **Samples prompt anchors** from high-norm textual keys to approximate the text-guided attention field.
2. **Selects prompt-aligned image patches** by keeping the strongest matches to each prompt anchor until a prompt budget \(K_p\) is filled.
3. **Covers the remaining visual manifold** via farthest point sampling (FPS), expanding the set to the global budget \(K\).
4. **Applies the same routine to Qwen2-VL** with configurable distance metrics and weighting \(\alpha\) to interpolate between prompt alignment and spatial diversity.

<p align="center">
  <img src="images/eta_distributions.png" alt="Distribution of adaptive pruning ratios across prompts" width="60%" />
</p>

MoB’s budgets and sampling rates are exposed through `MoB_config`, allowing practitioners to control where pruning happens (`l_idx`), how many tokens are kept (`K`), and how strongly the prompt objective is weighted.

### Integration into MLLMs
- **LLaVA / LLaVA-NeXT.** `LlamaModelMoB` subclasses the base decoder and slices hidden states before attention when the layer index matches `MoB_config['l_idx']`. The retained indices are cached for downstream logging or visualization.
- **Qwen2-VL.** `MoB` wraps the official Qwen2-VL model and injects the same balanced covering logic with cosine or Euclidean similarity options for prompt alignment.

---

## Quantitative & Qualitative Results

<p align="center">
  <img src="images/performance-efficiency.png" alt="Performance vs efficiency curve across benchmarks" width="85%" />
</p>

MoB attains state-of-the-art accuracy-efficiency trade-offs across strong- and weak-generation benchmarks, consistently outperforming naive top-k or entropy-based pruning baselines.

<p align="center">
  <img src="images/vis-strong.png" alt="Visualization of MoB retained patches on strong reasoning tasks" width="45%" />
  <img src="images/vis-weak.png" alt="Visualization of MoB retained patches on weak reasoning tasks" width="45%" />
</p>

<p align="center">
  <img src="images/samples.png" alt="Qualitative generation samples with MoB pruning" width="80%" />
</p>

<p align="center">
  <img src="images/ablation.png" alt="Ablation study on prompt and coverage budgets" width="70%" />
</p>

---

## Repository Structure
```
MoB/
├── LLaVA/                # MoB-enabled LLaVA backend (training, inference, evaluation)
├── LLaVA-NeXT/           # Extended LLaVA-NeXT models with MoB hooks
├── Qwen2-VL/             # Qwen2-VL integration with balanced covering decoder
├── lmms_eval/            # Evaluation toolkit instrumented for MoB logging
├── images/               # Figures used in the paper and README
└── README.md             # You are here
```

---

## Getting Started

### Environment Setup
1. **Clone the repo**
   ```bash
   git clone https://github.com/your-org/MoB.git
   cd MoB
   ```
2. **Create a Python environment** (Python ≥ 3.10 recommended):
   ```bash
   conda create -n mob python=3.10 -y
   conda activate mob
   ```

### Using MoB with LLaVA
1. **Install dependencies**
   ```bash
   cd LLaVA
   pip install -e .
   cd ..
   ```
   The package declares PyTorch 2.1.2, Hugging Face Transformers 4.37.2, and other required libraries for multimodal inference.
2. **Load a pretrained checkpoint with MoB enabled**
   ```python
   from llava.model.builder import load_pretrained_model

   tokenizer, model, image_processor, _ = load_pretrained_model(
       model_path="liuhaotian/llava-v1.6-vicuna-7b",
       model_base=None,
       model_name="llava-v1.6-mob"
   )

   model.config.MoB_config.update({
       "l_idx": 2,
       "image_token_start_index": 35,
       "image_token_length": 576,
       "K": 64,
       "Kp_ratios": 0.5,
       "k_ratio": 0.125,
       "prompt_samples": 8,
   })
   ```
   The default configuration above matches the setup we use in the paper.
3. **Run inference** using the standard LLaVA evaluation entry point:
   ```bash
   python -m llava.eval.run_llava \
       --model-path liuhaotian/llava-v1.6-vicuna-7b \
       --query "Describe the main differences between the two diagrams." \
       --image-file path/to/image.jpg
   ```
   MoB will prune image tokens automatically once `MoB_config` is set; retained patch IDs are stored on the model for downstream analysis.

### Using MoB with Qwen2-VL
1. **Install dependencies**
   ```bash
   cd Qwen2-VL
   pip install -r requirements.txt
   pip install -e qwen-vl-utils
   cd ..
   ```
   The requirement file covers streaming video support, FlashAttention (optional), and other vision-language utilities.
2. **Instantiate the MoB-wrapped model**
   ```python
   from Qwen2VL_MoB.modeling_qwen2_vl_self import MoB
   from transformers import AutoTokenizer, AutoConfig

   model_id = "Qwen/Qwen2-VL-7B-Instruct"
   tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
   config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
   config.MoB_config = {
       "image_token_start_index": 1,
       "image_token_length": 256,
       "reduction_ratio": 0.5,
       "prompt_ratio": 0.5,
       "alpha": 0.5,
       "distance_metric": "cosine",
   }
   model = MoB.from_pretrained(
       model_id,
       config=config,
       trust_remote_code=True,
       torch_dtype="auto"
   )
   ```
   You can tune `reduction_ratio`, `prompt_ratio`, and `alpha` to trade accuracy for efficiency at run time.
3. **Run multimodal generation** (example)
   ```python
   result = model.chat(tokenizer, query="What safety hazards does this blueprint reveal?", image=image_tensor)
   print(result)
   ```

### Benchmarking with lmms-eval
1. **Install the evaluation toolkit**
   ```bash
   cd lmms_eval
   pip install -e .
   cd ..
   ```
2. **Run a benchmark** (example: ScienceQA)
   ```bash
   lmms-eval \
       --model llava-v1.6-mob \
       --tasks scienceqa_img \
       --batch-size 1 \
       --log-retained-patches
   ```
   Our customized evaluator logs the indices of retained image tokens for every request, enabling detailed auditing of the pruning behavior.

---

## Model Zoo
| Backbone | Checkpoint | Notes |
|----------|------------|-------|
| LLaVA / LLaVA-NeXT | Coming soon | Release will include MoB-ready weights for 7B and 13B variants |
| Qwen2-VL | Coming soon | Inference-ready checkpoints with balanced covering defaults |

We will release links once the camera-ready version is finalized.

---

## Citation
If you find MoB useful for your research, please cite:

```bibtex
@inproceedings{li2024mob,
  title     = {Why 1 + 1 < 1 in Visual Token Pruning: Beyond Na{"i}ve Integration via Multi-Objective Balanced Covering},
  author    = {Li, Yangfu and colleagues},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```

---

## Acknowledgements & License
MoB builds on top of the open-source [LLaVA](https://github.com/haotian-liu/LLaVA) and [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) projects. All source code in this repository is released under the **MIT License**; please consult the `LICENSE` file for details.

Coming Soon.
