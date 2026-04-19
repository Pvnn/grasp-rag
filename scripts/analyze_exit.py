"""
analyze_exit.py

Step-by-step observability script for the EXIT compression pipeline.
Traces every stage with timing, token counts, batch shapes, and GPU memory
so you can see exactly where time is going.

Usage:
    python analyze_exit.py --token <HF_TOKEN>
    python analyze_exit.py --token <HF_TOKEN> --batch_size 4 --threshold 0.5
"""

import argparse
import time
import torch
import spacy
import textwrap
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── ANSI colors for readable terminal output ──────────────────────────────────
GRN  = "\033[92m"
YLW  = "\033[93m"
RED  = "\033[91m"
BLU  = "\033[94m"
CYN  = "\033[96m"
BOLD = "\033[1m"
RST  = "\033[0m"

def section(title: str):
    print(f"\n{BOLD}{BLU}{'─'*70}{RST}")
    print(f"{BOLD}{BLU}  {title}{RST}")
    print(f"{BOLD}{BLU}{'─'*70}{RST}")

def ok(msg):   print(f"  {GRN}✓{RST}  {msg}")
def warn(msg): print(f"  {YLW}⚠{RST}  {msg}")
def info(msg): print(f"  {CYN}→{RST}  {msg}")
def err(msg):  print(f"  {RED}✗{RST}  {msg}")

def gpu_stats() -> str:
    if not torch.cuda.is_available():
        return "CPU only"
    alloc  = torch.cuda.memory_allocated()  / 1024**3
    reserv = torch.cuda.memory_reserved()   / 1024**3
    total  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return (f"allocated={alloc:.2f}GB  reserved={reserv:.2f}GB  "
            f"total={total:.2f}GB  free={total-reserv:.2f}GB")

def fmt_tokens(n: int) -> str:
    return f"{n:,} tokens"

# ── Large synthetic corpus ─────────────────────────────────────────────────────
CORPUS = [
    {
        "title": "Transformer Architecture",
        "text": """
        The Transformer model was introduced by Vaswani et al. in 2017 in the paper
        "Attention Is All You Need". It replaced recurrent networks with self-attention
        mechanisms, allowing for much greater parallelism during training.
        The encoder processes the input sequence and produces a set of continuous
        representations. The decoder takes those representations and generates the output
        sequence one token at a time. Each encoder and decoder layer contains a
        multi-head self-attention sublayer and a position-wise feed-forward network.
        Residual connections and layer normalization are applied after each sublayer.
        The model uses learned positional encodings to inject information about the
        position of tokens in the sequence. Transformers have become the dominant
        architecture for natural language processing tasks.
        Pre-training on large corpora followed by fine-tuning has proven extremely effective.
        Models like BERT, GPT, and T5 are all based on the Transformer architecture.
        The attention mechanism computes a weighted sum of values based on query-key similarity.
        Multi-head attention allows the model to jointly attend to information from different
        representation subspaces at different positions.
        """,
    },
    {
        "title": "Retrieval-Augmented Generation",
        "text": """
        Retrieval-Augmented Generation (RAG) was proposed by Lewis et al. in 2020.
        It combines a parametric memory (the language model) with a non-parametric
        memory (a retrieval corpus) to improve knowledge-intensive NLP tasks.
        The retriever finds relevant documents from a large corpus given a query.
        The generator then conditions on both the query and the retrieved documents
        to produce the final answer.
        RAG models outperform purely parametric models on open-domain QA benchmarks.
        The retrieval component is typically a dense retriever like DPR.
        Dense Passage Retrieval (DPR) uses two BERT encoders: one for queries, one for passages.
        The top-k retrieved documents are passed to the generator as additional context.
        One challenge of RAG is that the retrieved context can be noisy or irrelevant.
        Context compression techniques like EXIT aim to filter out irrelevant sentences
        before passing the context to the generator, reducing token usage and noise.
        The kitchen sink is blue and the weather outside is stormy today.
        Compression reduces the number of tokens the reader model must process.
        This leads to lower inference latency and API costs in production deployments.
        """,
    },
    {
        "title": "Solid-State Drive Technology",
        "text": """
        Solid-state drives use NAND flash memory to persist data without moving parts.
        The absence of mechanical components means SSDs have much lower access latency
        than traditional hard disk drives, typically under 0.1 milliseconds.
        SSDs use a controller chip to manage read/write operations and wear leveling.
        Wear leveling distributes writes evenly across cells to extend drive lifespan.
        NVMe SSDs connect via the PCIe bus and offer far higher bandwidth than SATA SSDs.
        The price per gigabyte of NAND flash has fallen dramatically over the past decade.
        Enterprise SSDs include power-loss protection capacitors to prevent data corruption.
        TLC NAND stores three bits per cell, balancing capacity and cost.
        MLC NAND stores two bits per cell and offers better endurance than TLC.
        Boot times with an SSD are typically under 15 seconds compared to over a minute
        for a traditional hard drive. Application load times are similarly reduced.
        My cat knocked a glass of water off the counter this morning.
        Random read IOPS is the most relevant performance metric for typical workloads.
        """,
    },
    {
        "title": "Climate Change and Global Temperature",
        "text": """
        Global average surface temperature has increased by approximately 1.1 degrees
        Celsius above pre-industrial levels as of 2023, according to the IPCC.
        The primary driver of recent warming is the increase in atmospheric greenhouse
        gas concentrations, particularly CO2 from fossil fuel combustion.
        Arctic warming is occurring at roughly four times the global average rate,
        a phenomenon known as Arctic amplification.
        Melting ice sheets in Greenland and Antarctica are contributing to sea level rise.
        Extreme weather events including heatwaves, droughts, and intense precipitation
        are becoming more frequent and severe due to climate change.
        Ocean heat content has increased across all ocean basins.
        Coral bleaching events have become more frequent as sea surface temperatures rise.
        The Paris Agreement aims to limit global warming to 1.5 degrees above pre-industrial levels.
        Carbon capture and storage technologies may play a role in mitigation strategies.
        I had pasta for dinner last Tuesday and it was quite good.
        Renewable energy deployment, particularly solar and wind, has accelerated globally.
        """,
    },
    {
        "title": "Neural Network Training",
        "text": """
        Training a deep neural network requires a differentiable loss function and an
        optimization algorithm such as stochastic gradient descent or Adam.
        Backpropagation computes gradients of the loss with respect to all parameters
        by applying the chain rule of calculus recursively through the computation graph.
        Learning rate is one of the most important hyperparameters to tune.
        Too high a learning rate causes divergence; too low causes slow convergence.
        Learning rate schedules such as cosine annealing can improve final performance.
        Batch normalization stabilizes training by normalizing layer inputs.
        Dropout randomly sets activations to zero during training to reduce overfitting.
        Weight decay adds an L2 penalty on parameters to regularize the model.
        Mixed precision training uses float16 for forward and backward passes to reduce
        memory usage and speed up computation on modern GPUs with tensor cores.
        Gradient clipping prevents exploding gradients in recurrent models.
        Data augmentation artificially increases training set diversity.
        The sky turned an unusual shade of orange during the wildfire season.
        Early stopping monitors validation loss to prevent overfitting.
        """,
    },
]

QUERY = "How do solid-state drives improve computer performance and what makes them faster than hard drives?"


# ── Stage 1: Model loading ─────────────────────────────────────────────────────
def stage_model_loading(args):
    section("STAGE 1: Model Loading")

    t0 = time.perf_counter()
    info(f"Base model : google/gemma-2b-it")
    info(f"Checkpoint : doubleyyh/exit-gemma-2b")
    info(f"Quantization: 4-bit nf4 + double quant")
    info(f"GPU before load: {gpu_stats()}")

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2b-it", use_fast=True, token=args.token
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    ok(f"Tokenizer loaded  vocab_size={tokenizer.vocab_size:,}")

    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b-it",
        device_map="auto",
        quantization_config=qcfg,
        token=args.token,
    )
    ok(f"Base model loaded  {gpu_stats()}")

    model = PeftModel.from_pretrained(base, "doubleyyh/exit-gemma-2b", token=args.token)
    model.eval()

    device = next(model.parameters()).device
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("No",  add_special_tokens=False)[0]

    elapsed = time.perf_counter() - t0
    ok(f"PEFT model loaded  device={device}  time={elapsed:.2f}s")
    ok(f"Yes token id={yes_id}  No token id={no_id}")
    info(f"GPU after load : {gpu_stats()}")

    return model, tokenizer, device, yes_id, no_id


# ── Stage 2: spaCy sentence splitting ─────────────────────────────────────────
def stage_sentence_splitting():
    section("STAGE 2: Corpus Concatenation + Sentence Splitting")

    # Build full context exactly as compress() does
    t0 = time.perf_counter()
    context = "\n".join(
        f"{doc['title']}\n{doc['text']}" for doc in CORPUS
    )
    info(f"Full context length : {len(context):,} chars")

    nlp = spacy.load(
        "en_core_web_sm",
        disable=["tok2vec","tagger","parser","attribute_ruler","lemmatizer","ner"]
    )
    nlp.enable_pipe("senter")

    doc_spacy = nlp(context)
    sentences  = [s.text.strip() for s in doc_spacy.sents if s.text.strip()]
    elapsed    = time.perf_counter() - t0

    ok(f"Sentence splitting done  n_sentences={len(sentences)}  time={elapsed*1000:.1f}ms")

    print(f"\n  {'IDX':>3}  {'LEN':>5}  SENTENCE")
    print(f"  {'───':>3}  {'─────':>5}  {'─'*60}")
    for i, s in enumerate(sentences):
        preview = textwrap.shorten(s, width=60, placeholder="…")
        print(f"  {i:>3}  {len(s):>5}  {preview}")

    return sentences, context, nlp


# ── Stage 3: Tokenization analysis ────────────────────────────────────────────
def stage_tokenization(tokenizer, sentences, context, batch_size):
    section("STAGE 3: Prompt Tokenization Analysis (per batch)")

    def build_prompt(sentence):
        return (
            f'<start_of_turn>user\n'
            f'Query:\n{QUERY}\n'
            f'Full context:\n{context}\n'
            f'Sentence:\n{sentence}\n'
            f'Is this sentence useful in answering the query? '
            f'Answer only "Yes" or "No".<end_of_turn>\n'
            f'<start_of_turn>model\n'
        )

    batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]
    total_pad_tokens = 0
    total_real_tokens = 0

    print(f"\n  {'BATCH':>5}  {'SIZE':>4}  {'MIN_TOK':>7}  {'MAX_TOK':>7}  "
          f"{'PAD_TOK':>7}  {'PAD%':>5}  {'TOKENIZE_MS':>11}")
    print(f"  {'─────':>5}  {'────':>4}  {'───────':>7}  {'───────':>7}  "
          f"{'───────':>7}  {'────':>5}  {'───────────':>11}")

    for b_idx, batch in enumerate(batches):
        prompts = [build_prompt(s) for s in batch]

        t0 = time.perf_counter()
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            return_attention_mask=True,
        )
        tok_ms = (time.perf_counter() - t0) * 1000

        ids   = enc["input_ids"]           # [B, seq_len]
        mask  = enc["attention_mask"]      # [B, seq_len]
        lens  = mask.sum(dim=1).tolist()   # real tokens per item
        pad   = ids.numel() - mask.sum().item()
        pad_pct = 100 * pad / ids.numel() if ids.numel() else 0

        total_pad_tokens  += pad
        total_real_tokens += mask.sum().item()

        print(f"  {b_idx:>5}  {len(batch):>4}  {min(lens):>7}  {max(lens):>7}  "
              f"{pad:>7}  {pad_pct:>4.1f}%  {tok_ms:>9.1f}ms")

        if b_idx == 0:
            info(f"  First batch shape: {list(ids.shape)}  "
                 f"(batch_size × seq_len = {ids.shape[0]} × {ids.shape[1]})")
            if ids.shape[1] >= 4096:
                warn(f"  Sequence hit max_length=4096 — context is being TRUNCATED")

    total = total_pad_tokens + total_real_tokens
    pad_overall = 100 * total_pad_tokens / total if total else 0
    print()
    ok(f"Total real tokens : {fmt_tokens(total_real_tokens)}")
    warn(f"Total pad tokens  : {fmt_tokens(total_pad_tokens)}  ({pad_overall:.1f}% waste)")
    if pad_overall > 30:
        warn("High padding waste — sentences vary a lot in length within batches.")
        info("Consider sorting sentences by length before batching to reduce waste.")

    return batches, build_prompt


# ── Stage 4: Forward pass timing ──────────────────────────────────────────────
def stage_forward_passes(model, tokenizer, device, yes_id, no_id,
                         sentences, context, batch_size, build_prompt_fn):
    section("STAGE 4: Forward Pass Timing (full classification loop)")

    batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]

    results      = []   # (sentence, yes_prob, kept)
    batch_times  = []
    gpu_to_times = []
    infer_times  = []

    total_t0 = time.perf_counter()

    for b_idx, batch in enumerate(batches):
        prompts = [build_prompt_fn(s) for s in batch]

        # --- tokenize ---
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=4096, return_attention_mask=True,
        )

        # --- H→D transfer ---
        t_transfer = time.perf_counter()
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        torch.cuda.synchronize()
        transfer_ms = (time.perf_counter() - t_transfer) * 1000

        # --- forward pass ---
        t_infer = time.perf_counter()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(**enc)
                logits = out.logits[:, -1, :]
                rel    = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=1)
                probs  = torch.softmax(rel, dim=1)
                yes_p  = probs[:, 0].cpu().tolist()
        torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t_infer) * 1000

        batch_ms = transfer_ms + infer_ms
        batch_times.append(batch_ms)
        gpu_to_times.append(transfer_ms)
        infer_times.append(infer_ms)

        seq_len = enc["input_ids"].shape[1]
        print(f"  batch {b_idx:>2}  n={len(batch)}  seq={seq_len:>4}  "
              f"transfer={transfer_ms:>6.1f}ms  infer={infer_ms:>7.1f}ms  "
              f"total={batch_ms:>7.1f}ms  |  gpu: {gpu_stats()}")

        for sent, yp in zip(batch, yes_p):
            results.append((sent, yp, yp >= 0.5))

    total_ms = (time.perf_counter() - total_t0) * 1000

    section("STAGE 4 SUMMARY")
    ok(f"Total sentences classified : {len(sentences)}")
    ok(f"Total batches              : {len(batches)}")
    ok(f"Total wall time            : {total_ms:.1f}ms  ({total_ms/1000:.2f}s)")
    ok(f"Avg time per batch         : {sum(batch_times)/len(batch_times):.1f}ms")
    ok(f"Avg H→D transfer per batch : {sum(gpu_to_times)/len(gpu_to_times):.1f}ms")
    ok(f"Avg forward pass per batch : {sum(infer_times)/len(infer_times):.1f}ms")
    ok(f"Avg time per sentence      : {total_ms/len(sentences):.1f}ms")

    fwd_pct = 100 * sum(infer_times) / total_ms
    xfr_pct = 100 * sum(gpu_to_times) / total_ms
    info(f"Time breakdown  →  forward={fwd_pct:.1f}%  transfer={xfr_pct:.1f}%  other={100-fwd_pct-xfr_pct:.1f}%")

    if sum(infer_times) / len(infer_times) > 2000:
        warn("Forward passes are very slow — likely due to 4-bit dequantization overhead on a small GPU.")
        warn("On a 3090 with fp16 (no quantization) these should be 3-5× faster.")

    return results


# ── Stage 5: Classification results ───────────────────────────────────────────
def stage_results(results, sentences):
    section("STAGE 5: Classification Results")

    kept    = [(s, p) for s, p, k in results if k]
    dropped = [(s, p) for s, p, k in results if not k]

    print(f"\n  {'IDX':>3}  {'PROB':>5}  {'KEEP':>4}  SENTENCE")
    print(f"  {'───':>3}  {'─────':>5}  {'────':>4}  {'─'*60}")
    for i, (sent, prob, keep) in enumerate(results):
        flag    = f"{GRN}YES{RST}" if keep else f"{RED} NO{RST}"
        preview = textwrap.shorten(sent, width=60, placeholder="…")
        print(f"  {i:>3}  {prob:.3f}  {flag}  {preview}")

    print()
    ok(f"Sentences kept    : {len(kept)}/{len(sentences)}")
    ok(f"Sentences dropped : {len(dropped)}/{len(sentences)}")
    ok(f"Compression ratio : {len(kept)/len(sentences)*100:.1f}% of sentences retained")

    print(f"\n  {BOLD}Kept sentences:{RST}")
    for s, p in kept:
        print(f"    [{p:.3f}] {textwrap.shorten(s, 80, placeholder='…')}")

    print(f"\n  {BOLD}Dropped sentences:{RST}")
    for s, p in dropped:
        print(f"    [{p:.3f}] {textwrap.shorten(s, 80, placeholder='…')}")

    # Sanity check — are obviously irrelevant sentences being filtered?
    irrelevant_keywords = ["cat", "pasta", "kitchen sink", "orange", "water"]
    for s, p, k in results:
        for kw in irrelevant_keywords:
            if kw in s.lower() and k:
                warn(f"Irrelevant sentence KEPT (score={p:.3f}): {textwrap.shorten(s, 60)}")


# ── Stage 6: End-to-end latency vs naive per-doc ──────────────────────────────
def stage_latency_comparison(total_sentences, batch_times, batch_size):
    section("STAGE 6: Latency Comparison — Batched vs Naive Per-Doc")

    n_docs         = len(CORPUS)
    avg_batch_ms   = sum(batch_times) / len(batch_times) if batch_times else 0
    batched_total  = sum(batch_times)

    # Naive estimate: each sentence gets its own forward pass (old per-doc behavior)
    naive_total = total_sentences * avg_batch_ms   # each call has batch_size=1 avg overhead

    info(f"Corpus              : {n_docs} documents  {total_sentences} sentences")
    info(f"Batch size          : {batch_size}")
    info(f"Batched total       : {batched_total:.1f}ms ({batched_total/1000:.2f}s)")
    info(f"Naive per-sentence  : ~{naive_total:.1f}ms ({naive_total/1000:.2f}s)  [estimated]")

    if naive_total > batched_total:
        speedup = naive_total / batched_total
        ok(f"Batching speedup    : ~{speedup:.1f}x")
    else:
        warn("Batched is not faster — possible tokenization or padding overhead dominating.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",      type=str, required=True, help="HuggingFace token")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold",  type=float, default=0.5)
    args = parser.parse_args()

    print(f"\n{BOLD}EXIT Pipeline Analyzer{RST}")
    print(f"Query      : {QUERY}")
    print(f"Corpus     : {len(CORPUS)} documents")
    print(f"Batch size : {args.batch_size}")
    print(f"Threshold  : {args.threshold}")
    print(f"CUDA       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")

    model, tokenizer, device, yes_id, no_id = stage_model_loading(args)

    sentences, context, nlp = stage_sentence_splitting()

    batches, build_prompt_fn = stage_tokenization(
        tokenizer, sentences, context, args.batch_size
    )

    results = stage_forward_passes(
        model, tokenizer, device, yes_id, no_id,
        sentences, context, args.batch_size, build_prompt_fn
    )

    batch_times_ms = []   # re-collect from stage 4 summary for comparison
    stage_results(results, sentences)

    stage_latency_comparison(len(sentences), [], args.batch_size)

    section("DONE")
    ok("All stages complete. Check warnings above for bottlenecks.")


if __name__ == "__main__":
    main()