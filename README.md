# Tweet Like Me

Tweet Like Me is an end-to-end pipeline that given a specified twitter user, will download their tweets, format them, train a model, and generate texts meant to simulate the user's tweets. The model is an RNN architecture that uses the user's tweets as the corpus for training the Word2Vec embeddings.

### Prerequisites

In order to run this project, one requires to have API credentials for Twitter's API. The credentials will then be entered into config.py

### Installing

Download the repository

```
git clone https://github.com/lccuellar08/TweetLikeMe.git
```

Insall the requirements

```
pip install requirements.txt
```

## Running

To run the end to end pipeline, all you need is the Twitter handle of a publicly available user, and Twitter API credentials.

First enter the credentials into config.py

Then we run main.py using the Twitter handle as an argument

```
python main.py barackobama
```

## Acknowledgments

* Tweet_Downloader.py uses code from https://gist.github.com/yanofsky/5436496
