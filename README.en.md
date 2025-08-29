
# About the source code

I’m an independent researcher working on this in my spare time, since I have a full-time job and limited availability. I thought the paper would be on hold for a while, and I originally planned to spend the next few days preparing the code repository. Unexpectedly, after submitting to arXiv yesterday, the paper was officially published today.

The source code will be released shortly.

# How Was TLinFormer Born? A Journey of Thought
We begin with a problem common in sparse attention mechanisms: information loss.

In traditional sparse methods, the model computes attention scores for all tokens, then ruthlessly discards those with lower scores. While efficient, the cost is that the information carried by those discarded tokens may be permanently lost—a potentially fatal issue for tasks requiring long-range dependencies.

## Step One: An attempt at sparsification without discarding any information

To address this problem, my initial version took a gentler approach. I also selected the Top-K tokens with the highest attention scores—the “top students”—but I did not discard the other “ordinary” tokens. Instead, I let these “top students” (Top-K tokens) perform another round of cross-attention with all tokens (the global information), thereby integrating the entire history into themselves.
In this way, we gained the computational efficiency of sparsification without crudely discarding any information.

## Step Two: A fundamental question

At this point, a critical question arose in my mind: what is the essence of this approach?

Since these selected “top students” ultimately have to turn back and integrate information from everyone else, is our painstaking selection of “top students” still meaningful? If the neural network is powerful enough, wouldn’t it be able to learn, through parameter optimization, how to extract everything it needs from the global information even if I chose the Top-K tokens randomly?

This realization led me to see that the essence of the attention mechanism can actually be viewed as a highly flexible, learnable form of full connectivity along the sequence length (L) dimension.

## Step Three: Inspiration from the principle of locality

Then, I drew inspiration from a classic concept in computer science—the principle of locality. Programs, when executed, tend to access nearby data in both time and space. For a Transformer, could it be that the tokens nearest to the current output window are naturally more important?

The answer is yes. So I abandoned the idea of globally searching for “top students” and instead directly chose the Top-K tokens closest to the current output window. This was not only more logical, but also more computationally efficient, and ultimately gave rise to TLinFormer’s distinctive dual-window structure.

## Conclusion: A smarter form of sparsification

From this perspective, TLinFormer is also a kind of sparsification solution. But it is more refined, because it hands over all the optimization work entirely to the neural network itself.

It no longer requires us to design complicated rules to “guess” which information is important, nor does it risk discarding any information. It simply provides a simple, efficient structure based on the principle of locality, and then lets the model learn during training how to best exploit both local and global information.