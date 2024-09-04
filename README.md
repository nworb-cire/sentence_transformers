# Sentence Transformers

This repository contains code for lightweight sentence transformers to be used for NLP tasks. This code was developed as part of a take-home interview project and should not be used in a production environment. The code is optimized for readability and simplicity, not for speed or efficiency. The training code is minimal and has not been tested on large datasets.


## Usage
I have included a Dockerfile which you can build and run to use the code from step 2.

# Step 1: Implement a Sentence Transformer Model
You can see the code for this task in `encoder.py`. I chose to use a very simple, no-frills BERT-like architecture. This is partly out of time constraint, and partly because I believe it is ultimately better to use or fine-tune a pre-trained model for such tasks than it is to build one from scratch. If someone else has already done the heavy lifting, why reinvent the wheel?

# Step 2: Multi-Task Learning Expansion
You can see the code for this task in `multi_task.py`. For the encoder, instead of reusing my work from part 1 I chose to use a pre-trained encoder model. I use [GIST-small-Embedding-v0](https://github.com/worldbank/GISTEmbed) which is the best ranked model on the [Massive Text Embedding Benchmark Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for English language classification tasks under 100M parameters. I do this because I believe that the pre-trained model will be more performant and more effective than the simple model I implemented in part 1. I implement two tasks: a binary classification head and a NER head. The binary classification head is a simple logistic regression layer on top of the sentence embeddings. The NER head is a simple linear layer on top of the sentence embeddings. If a more complex model is needed, one or both of these could be replaced with a non-linear model.

# Step 3: Discussion

#### Consider the scenario of training the multi-task sentence transformer that you implemented in Task 2. Specifically, discuss how you would decide which portions of the network to train and which parts to keep frozen. 

The decision of which components to train and which to freeze depends heavily on the task at hand. Some factors to consider are the dataset size, the similarity of tasks, and the computational resources available.

If the task is not a real-time streaming application, I would propose finding the most relevant, best performing pre-trained model on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) to generate embeddings for the whole dataset and save them off. Then I would train task-specific heads on top of the embeddings. This approach is efficient and allows for easy experimentation with different tasks and models. It also works well on small datasets, especially if the task-specific heads are simple such as logistic layers.

If the task is a streaming application, I would still propose doing the above, but with a smaller pre-trained model that can be run in real-time. It is important to stress test the throughput of such models and potentially configure infrastructure to scale up based on demand.

If the dataset is very large and resources are abundant, training a full model end-to-end will likely yield the best results. This approach is more computationally intensive and requires significant expertise to do well, so I would advise against it if you want to move quickly.

If you have a highly imbalanced dataset for your two tasks, you may want to consider transfer learning techniques such as copying parameters from one head to another and fine tuning (if the tasks are similar) or sharing most layers of a nonlinear model and training the two final layers separately, or with different learning rates.

#### Discuss how you would decide when to implement a multi-task model like the one in this assignment and when it would make more sense to use two completely separate models for each task. 

So long as the tasks are similar enough, it makes sense to use a multi-task model. Even if the tasks are different, if you can train a large model end-to-end, then multi-task learning may still be beneficial and generalize better.

It makes sense to use separate models when the tasks are different and performance is paramount. For instance, the best pre-trained model for classification may have much better performance than the best pre-trained model for retrieval, with no happy medium. In this case, it makes sense to use separate models. It may also make sense to use different models if inference is not done concurrently, as you can then scale each model independently, and you don't need to have one large model running all the time.

#### When training the multi-task model, assume that Task A has abundant data, while Task B has limited data. Explain how you would handle this imbalance.

I would want to do an in-depth data analysis to understand why the datasets are imbalanced. Why is dataset B smaller? Is this size difference caused statistical bias? If so, it becomes far more tricky to work with. For instance, in my past job, our model played the role of gatekeeper, so we only had outcomes for the cases that were approved by the model. This caused a statistical bias in the dataset and the inability to evaluate false positives. 

That said, there are a few options to try out. As with any machine learning problem, it is important to experiment and see what works best on a holdout test set.

1. Weight the loss function to counteract the dataset imbalance. This is a good choice if the smaller dataset is a representative sample. 
2. If using a shared architecture, learn task A first, then freeze everything but head B.
3. If possible, synthetic data could be generated for task B. This is heavily dependent on what task B is and may not be feasible.
4. Asymmetric regularization techniques could be used to prevent overfitting on task A, such as a higher dropout rate or L2 regularization on the task A head.

