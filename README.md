## Docker Container for Embedding Visualizer
The embedding visualizer is a small tool to visualize semantic similarity of sentences or words to each other, using cosine similarity. For more details, see the [frontend](https://github.com/ziHanDing579/embed-visual) and the [terraform IaC](https://github.com/ziHanDing579/embedLambda). The live site is at [zihanding579.github.io/embed-visual](https://zihanding579.github.io/embed-visual/).

This repo contains the dockerfile and the python code needed to export and run the model.

In this README, I will go over the design decisions behind the container itself.

### Why MiniLM?
There are many different embedding models out there, notably, the `bge-large-en` or `qwen-embedding` or many others on the MTEB leaderboards. `MiniLM` stands out as it is not only one of the most well-known and proven models but also small. Our task requires something to run on the CPU with limited memory, so `MiniLM` seem to match the criteria perfectly. If we need to, upgrading it from MiniLM to any other model is also fairly simple. Finally, another consideration is that I used `MiniLM` in my own work, in both DaMiT-SQL and my thesis.

### Why ONNX runtime?
I did not know about ONNX runtime before this project. Mostly because CPU inference with limited resources was not the chief concern in my work. However, for this demo tool that is meant to run on AWS Lambda, I had to consider the cold start and limited storage capacity. This meant I had to consider Pytorch and sentence-transformers, both of which brings in massive dependencies that bloats the size of the container to well above 1 GB (though I am aware that we have 10 GB).

ONNX runtime helps to do two things - one, reduce package size and two, reduce CPU inference time. Unlike a all-in-one framework like Pytorch, ONNX runtime is specialized and perfect for my use case. No training needed, just running the model. It also supports easy quantization (**one** additional argument for the architecture) so if I do need to switch `MiniLM` to something bigger, there would be minimal changes to the code.

### ECR and SSM
Given that the project runs on AWS Lambda, the container needed to go to the ECR. For this repo, I chose to go with a manually created ECR role and policy instead of IaC. I could have used terraform, but it adds an additional layer when I've already split the responsibilities by having a container repo.

As for the SSM to store my tags, the main reason is that I would like the Terraform pipeline to be more automated. Having a `latest` tag would mean that Terraform cannot actively detect a change in the image, and hardcoding the value goes against IaC principles. So SSM and SHA hashes as tags seemed the natural choice.