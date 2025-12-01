This is Melvin, an AI Machine Learning Engineer. Inspired from Devin which is an AI Software Engineer. Below are the things that Melvin can do autonomously: 
- Analyse the MLEbench lite datasets from Kaggle.
- Decides the type of ML task it needs to perform based on the analysis.
- Generates the optimal training script for the dataset.
- Runs the training script.
- Generates a submission.csv.
- Uses MLEbench's grader tool to get a score.


## Steps for Reproducability
1. Make a parent folder. Name it whatever you want. Eg: `mkdir MLEagent`
2. cd `MLEagent`
3. git clone `https://github.com/rajarshiroydev/melvin.git`
4. git clone `https://github.com/openai/mle-bench.git`

After doing the first 4 steps, the folder should look like this:
```
MLEagent/
├── melvin/
└── mle-bench/
```
5. cd `melvin`
6. cp `.env.example` `.env`
7. Paste your gemini api key in `.env`
8. Start two terminals. One in melvin and the other in mle-bench.
9. In both the terminals, do `uv venv` then `source .venv/bin/activate` then `uv sync`. 

NOTE: Sometimes the IDE doesn't pick up the correct interpreter path so do `which python` while the virutal environment is active, copy that path. Do `cmd + shift + p`, (VScode) search for _python select interpreter_ then paste the interpreter path that you had copied. That should resolve any missing import errors.

10. Now that you have both the terminals ready. Go to `mle-bench` directory and download any dataset that you want from the MLEbench datasets list using `mlebench prepare -c <competition-id>`. competition-id is simply the competition name. You need to follow step 11 before using the command.
11. You also need a _kaggle.json_ file saved in your .kaggle folder in order to be able to download the dataset with the above mentioned command. For this, go to your Kaggle profile settings, click on _Create Legagy API Key_, a _kaggle.json_ file will be downloaded. Place it in your .kaggle folder.
12. Once you have downloaded your desired dataset, go to the melvin terminal and use the command `python agents/orchestrator.py -c <competition-id>`. This will initiate the agent to perform its ML Engineering tasks.