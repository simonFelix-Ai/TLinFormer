
# Design Philosophy

Does a model truly need an infinite, lossless memory? Or is learning to "summarize" and "forget" an essential pathway to higher intelligence?

I did not want to sacrifice any historical information (Full Context Awareness) or the precision of attention calculations (Exact Attention), yet I still aimed for linear computational complexity (Linear Complexity).

Instead of following the mainstream paths of "approximating Softmax" (Kernel Methods) or "discarding tokens" (Sparse Methods), TxFormer returns to the level of the neural network's "connection topology" with first-principles thinking, solving the efficiency problem by reconstructing the flow of information.

TxFormer is more like a brilliant thought experiment exploring the ultimate possibilities of efficiency, rather than a general-purpose solution ready for large-scale industrial adoption.

It intentionally introduces an information bottleneck, trading lossy compression for theoretically extreme efficiency: TconstFormer theoretically achieves amortized O(1) computation and strict O(1) memory.

# An Encoder-Decoder Perspective

TxFormer's dual-window structure is, in essence, a form of encoder-decoder architecture. It brings the encoder-decoder architecture back into the realm of Large Language Models (LLMs). Unlike a standard transformer, which fully encodes the entire history before feeding it to the decoder, TxFormer adopts a strategy of layered interaction between the encoder and decoder.

# Limitations
As the author truly lacks the financial resources and energy to verify the model's performance with a large number of parameters, and considering its philosophical design of forced compression, its performance on "needle in a haystack" tasks that require precise retrieval remains an open question.

Theoretically, TConstFormer's O(1) KV cache and computation are more elegant and disruptive, whereas TLinFormer's O(N) is merely an improvement upon existing linear attention methods. In practice, however, TLinFormer may be a more balanced and practical architecture than TConstFormer. Its information compression is less extreme, preserving more potential for direct interaction with a long history, likely achieving a better trade-off between performance and efficiency.

# The Origin of TLinFormer/TconstFormer

We begin with a problem that is prevalent in sparse attention mechanisms: information loss.

In traditional sparse methods, the model calculates attention scores for all global tokens and then unfeelingly discards those with lower scores. While this approach is efficient, the cost is that the information carried by these discarded tokens may be permanently lost, which can be fatal for tasks requiring long-range dependencies.

## Step 1: An Attempt at Sparsity with "Zero Information Loss"
To solve this problem, my initial version took a gentler route. I would still select the Top-K tokens with the highest attention scores—the "top students"—but I wouldn't discard the other "ordinary" tokens. Instead, I would have these "top students" (the Top-K tokens) perform another round of cross-attention with all tokens (the global context), allowing them to fuse the complete history into themselves.

This way, we could achieve the computational efficiency of sparsity without crudely discarding any information.

## Step 2: A Question That Strikes at the Essence
But at this point, a key question emerged in my mind: What is the essence of this process?

Since the selected "top students" ultimately have to go back and fuse information from everyone, is the painstaking process of selecting them truly that meaningful? If the neural network is powerful enough, could I just randomly select K tokens and, through parameter optimization, it would learn how to extract everything it needs from the global context?

This thought led me to realize that the essence of the attention mechanism can be seen as a highly flexible, learnable fully-connected layer across the sequence length (L) dimension.

## Step 3: Drawing Inspiration from the "Principle of Locality"

From there, I drew inspiration from a classic concept in computer science—the "principle of locality." When a program executes, it tends to access data that is nearby, both in time and space. So, for a Transformer, aren't the tokens closest to the current output window naturally more important?

Therefore, I abandoned the idea of a global search for "top students" and instead directly selected the Top-K tokens closest to the current output window. This ultimately leading to the unique dual-window structure of TLinFormer today.

## Conclusion: A Smarter Sparsity
From this perspective, TLinFormer is indeed a solution for sparsity. But it is a more intelligent one because it delegates all the optimization work entirely to the neural network itself.

It no longer requires us to design complex rules to "guess" which information is important, nor does it risk discarding any information. It simply provides a simple, efficient structure based on the "principle of locality" and lets the model learn on its own during training how to best utilize this local and global information.

# Future Plans

I am not sure whether to focus on foundational architecture or on applications. Pursuing foundational architecture would next lead to continual learning, and I already have some vague ideas for implementation paths. However, working on the low-level infrastructure is exhausting and offers no positive financial incentives. Even a neural network can't perform reinforcement learning without positive reinforcement. On the other hand, pursuing applications would require building a product, which could generate some income. But for applications, I lack sufficient computational power and energy. A model doesn't exhibit emergent abilities until its parameter count reaches a certain scale. To increase the parameter count, say to 1 billion, the cost of cloud GPUs for training becomes a significant burden that, given my personal financial situation, is difficult to bear.For now, it feels like a dead end, and I don't know what to do. Fortunately, this is just a hobby project, so it doesn't interfere with my day job.
